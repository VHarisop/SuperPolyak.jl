using LinearAlgebra
using Printf
using SparseArrays

import ReverseDiff: compile, gradient!, GradientTape
import QPSReader: readqps
import SCS: IndirectSolver, DirectSolver, scs_solve

import SuperPolyak

struct QuadraticProgram
  A::AbstractMatrix{Float64}
  P::AbstractMatrix{Float64}
  b::Vector{Float64}
  c::Vector{Float64}
end

"""
  read_problem_from_qps_file(filename::String, mpsformat; no_eqs::Bool = true) -> QuadraticProgram

Read a quadratic programming problem from a .QPS file and convert to the
representation used by the SCS solver.
"""
function read_problem_from_qps_file(filename::String, mpsformat; no_eqs::Bool = true)
  problem = readqps(filename, mpsformat=mpsformat)
  m, n = problem.ncon, problem.nvar
  # The objective matrix is symmetric and the .QPS file gives
  # the lower-triangular part only.
  P = sparse(problem.qrows, problem.qcols, problem.qvals, n, n)
  P = P + tril(P, 1)'
  A = sparse(problem.arows, problem.acols, problem.avals, m, n)
  ℓ = problem.lcon
  u = problem.ucon
  # Ax + s = b; s ≥ 0.
  A_aug = [-A; A; -sparse(1.0I, n, n); sparse(1.0I, n, n)]
  b_aug = [-ℓ; u; -problem.lvar; problem.uvar]
  # Only include nontrivial variable bounds (SCS will yield nan's otherwise).
  finite_ind = isfinite.(b_aug)
  return QuadraticProgram(
    A_aug[finite_ind, :],
    copy(P),
    b_aug[finite_ind],
    problem.c,
  )
end

struct ScsResult
  sol::Vector{Float64}
  iter::Int
  solve_time::Float64
  scale::Float64
  status::String
  status_val::Int
end

"""
  solve_with_scs(qp::QuadraticProgram,
                 primal_sol::Vector{Float64},
                 dual_sol::Vector{Float64},
                 slack::Vector{Float64},
                 ϵ_tol::Float64)

Solve a `QuadraticProgram` with SCS up to tolerance `ϵ_tol`, starting from a
solution `(primal_sol, dual_sol, slack)`.
"""
function solve_with_scs(
  qp::QuadraticProgram,
  primal_sol::Vector{Float64},
  dual_sol::Vector{Float64},
  slack::Vector{Float64};
  ϵ_tol::Float64,
  ϵ_rel::Float64,
  use_direct_solver::Bool = true,
  initial_scale::Float64 = 0.1,
  iteration_limit::Int = 100000,
)
  m, n = size(qp.A)
  result = scs_solve(
    use_direct_solver ? DirectSolver : IndirectSolver,
    m,
    n,
    qp.A,
    triu(qp.P),
    qp.b,
    qp.c,
    0,                # number of equalities
    m,                # number of inequalities
    zeros(0),
    zeros(0),
    zeros(Int, 0),
    zeros(Int, 0),
    0,
    0,
    zeros(0),
    primal_sol,
    dual_sol,
    slack,
    warm_start = true,
    eps_abs = ϵ_tol,
    eps_rel = ϵ_rel,
    eps_infeas = 1e-100,
    scale = initial_scale,
    max_iters = max(iteration_limit, 1),
  )
  solve_status = (result.info.status_val in [1; 2]) ? "SOLVED" : "UNSOLVED"
  return ScsResult(
    [result.x; result.y; result.s],
    result.info.iter,
    1e-3 * result.info.solve_time,
    result.info.scale,
    solve_status,
    result.info.status_val,
  )
end

"""
  proj_cone!(qp::QuadraticProgram, y::Vector{Float64}, s::Vector{Float64})

Project `(s, y)` to the cone `K × K*` specified by a `QuadraticProgram`.
"""
function proj_cone!(
  qp::QuadraticProgram,
  z::Vector{Float64},
)
  m, n = size(qp.A)
  y = view(z, (n+1):(n+m))
  s = view(z, (n+m+1):(n+2m))
  both_positive = (y .> 0) .& (s .> 0)
  # s, y should be nonnegative and s_i y_i = 0.
  s[s .< 0] .= 0.0
  y[y .< 0] .= 0.0
end

"""
  kkt_error(qp::QuadraticProgram)

Return a callable that computes the KKT error of a `QuadraticProgram`.
"""
function kkt_error(qp::QuadraticProgram)
  A = qp.A
  b = qp.b
  c = qp.c
  P = qp.P
  m, n = size(A)
  loss_fn(z::AbstractVector) = begin
    x = z[1:n]
    y = z[(n+1):(n+m)]
    s = z[(n+m+1):end]
    Px = P * x
    return max(
      norm(A * x - b, Inf),
      norm(Px + A'y + c, Inf),
      abs(x' * Px + c'x + b'y),
      norm(s .* y, Inf),
    )
  end
  return loss_fn
end

"""
  forward_backward_error(qp::QuadraticProgram)

Return a callable that computes the forward-backward error for a
`QuadraticProgram`.
"""
function forward_backward_error(qp::QuadraticProgram)
  A = qp.A
  b = qp.b
  c = qp.c
  P = qp.P
  n = size(A, 2)
  # Expect length(z) = n + m.
  loss_fn(z::AbstractVector) = begin
    x = z[1:n]
    y = z[(n+1):end]
    diff_x = c + P*x + A'y      # x - (x - (c + Qx + A'y))
    diff_y = y - max.(0.0, y - (b - A * x))
    return max(norm(diff_x, Inf), norm(diff_y, Inf))
  end
  return loss_fn
end

"""
  forward_backward_error_subgradient(qp::QuadraticProgram)

Return a callable that computes a subgradient of the forward-backward error for
a `QuadraticProgram`.
"""
function forward_backward_error_subgradient(qp::QuadraticProgram)
  m, n = size(qp.A)
  compiled_loss_tape = compile(GradientTape(forward_backward_error(qp), randn(m + n)))
  return z -> gradient!(compiled_loss_tape, z)
end

function kkt_error_subgradient(qp::QuadraticProgram)
  m, n = size(qp.A)
  compiled_loss_tape = compile(GradientTape(kkt_error(qp), randn(2m + n)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  fallback_algorithm(qp::QuadraticProgram, x₀::Vector{Float64}, y₀::Vector{Float64},
                     exit_frequency::Int, oracle_calls_limit::Int, ϵ_tol::Float64,
                     ϵ_rel::Float64, scale::Float64)

The fallback algorithm used when solving a QP with forward-backward error as
the loss function.
"""
function fallback_algorithm(
  qp::QuadraticProgram,
  x₀::Vector{Float64},
  y₀::Vector{Float64},
  exit_frequency::Int,
  oracle_calls_limit::Int,
  ϵ_tol::Float64,
  ϵ_rel::Float64,
  scale::Float64,
)
  loss_fn = forward_backward_error(qp)
  m, n = size(qp.A)
  iter_total = 0
  time_total = 0.0
  z_prev = zeros(n + 2m)
  copyto!(z_prev, 1, x₀, 1, n)
  copyto!(z_prev, n+1, y₀, 1, m)
  # Determine s from x₀ and y₀.
  z_prev[(n+m+1):end] .= max.(qp.b - qp.A * x₀, 0.0)
  while iter_total < oracle_calls_limit
    scs_result = solve_with_scs(
      qp,
      z_prev[1:n],
      z_prev[(n+1):(n+m)],
      z_prev[(n+m+1):end],
      ϵ_tol = ϵ_tol,
      ϵ_rel = ϵ_rel,
      use_direct_solver = true,
      iteration_limit = exit_frequency,
      initial_scale = scale,
    )
    iter_total += scs_result.iter
    time_total += scs_result.solve_time
    scale = scs_result.scale
    if (scs_result.status_val in [1; 2])
      # Warm-start.
      copyto!(z_prev, scs_result.sol)
    else
      # Exit unsuccessfully if solver status was not 'Solved'.
      return ScsResult(
        scs_result.sol,
        iter_total,
        time_total,
        scale,
        "UNSOLVED",
        scs_result.status_val,
      )
    end
    # Keep z[1:(n+m)] = (x, y) to evaluate loss.
    new_loss = loss_fn(z_prev[1:(n+m)])
    @info "Fallback - it = $(iter_total) - loss = $(@sprintf("%.8f", new_loss))"
    # Exit successfully if forward-backward was sufficiently reduced.
    if new_loss ≤ ϵ_tol
      return ScsResult(
        z_prev,
        iter_total,
        time_total,
        scale,
        "SOLVED",
        1,        # Status value for 'Solved'.
      )
    end
  end
end

"""
  superpolyak_with_scs(qp::QuadraticProgram, z₀::Vector{Float64};
                       ϵ_tol::Float64 = 1e-15, ϵ_rel::Float64 = 1e-15,
                       ϵ_decrease::Float64 = 1/2, ϵ_distance::Float64 = 3/2,
                       η_est::Float64 = 1.0, η_lb::Float64 = 0.1,
                       exit_frequency::Int = 250, oracle_calls_limit::Int = 100000,
                       bundle_budget::Int = length(z₀), kwargs...)

Run SuperPolyak using the SCS solver as the fallback method to solve a
`QuadraticProgram`. The loss used is the forward-backward error.
"""
function superpolyak_with_scs(
  qp::QuadraticProgram,
  z₀::Vector{Float64};
  ϵ_tol::Float64 = 1e-15,
  ϵ_rel::Float64 = 1e-15,
  ϵ_decrease::Float64 = (1 / 2),
  ϵ_distance::Float64 = (3 / 2),
  η_est::Float64 = 1.0,
  η_lb::Float64 = 0.1,
  exit_frequency::Int = 250,
  oracle_calls_limit::Int = 100000,
  bundle_budget::Int = length(z₀),
  kwargs...,
)
  f = forward_backward_error(qp)
  g = forward_backward_error_subgradient(qp)
  if (ϵ_decrease ≥ 1) || (ϵ_decrease < 0)
    throw(BoundsError(ϵ_decrease, "ϵ_decrease must be between 0 and 1"))
  end
  if (ϵ_decrease * ϵ_distance > 1)
    throw(
      BoundsError(
        ϵ_decrease * ϵ_distance,
        "ϵ_decrease * ϵ_distance must be < 1",
      ),
    )
  end
  z = z₀[:]
  fvals = [f(z₀)]
  oracle_calls = [0]
  elapsed_time = [0.0]
  step_types = ["NONE"]
  idx = 0
  # Initial dual scale factor for calls to SCS.
  scs_scale = 0.1
  # Number of variables and constraints.
  n = size(qp.A, 2)
  @info "Using bundle system solver: QR_COMPACT_WV"
  while true
    cumul_time = 0.0
    Δ = fvals[end]
    η = ϵ_distance^(idx)
    target_tol = max(ϵ_decrease * Δ, ϵ_tol)
    # Remaining budget for the bundle method.
    budget_rem = min(bundle_budget, oracle_calls_limit - sum(oracle_calls))
    bundle_stats = @timed bundle_step, bundle_calls =
      SuperPolyak.build_bundle_wv(f, g, z, η, 0.0, η_est, budget_rem)
    cumul_time += bundle_stats.time - bundle_stats.gctime
    # Adjust η_est if the bundle step did not satisfy the descent condition.
    bundle_loss = isnothing(bundle_step) ? Δ : f(bundle_step)
    if !isnothing(bundle_step) && (bundle_loss > Δ^(1 + η_est)) && (Δ < 0.5)
      η_est = max(η_est * 0.9, η_lb)
      @debug "Adjusting η_est = $(η_est)"
    end
    if isnothing(bundle_step) || (bundle_loss > target_tol)
      if (!isnothing(bundle_step)) && (bundle_loss < Δ)
        copyto!(z, bundle_step)
      end
      @info "Bundle step failed (k=$(idx), f=$(bundle_loss)) -- using fallback algorithm"
      scs_result = fallback_algorithm(
          qp,
          z[1:n],
          z[(n+1):end],
          exit_frequency,
          # oracle_calls_limit
          max(oracle_calls_limit - (sum(oracle_calls) + bundle_calls), 1),
          # ϵ_tol
          target_tol,
          ϵ_rel,
          # initial_scale
          scs_scale,
        )
      copyto!(z, scs_result.sol)
      fallback_calls = scs_result.iter
      @info "Updating scs scale: $(scs_result.scale) from $(scs_scale)"
      scs_scale = scs_result.scale
      cumul_time += scs_result.solve_time
      # Include the number of oracle calls made by the failed bundle step.
      push!(oracle_calls, fallback_calls + bundle_calls)
      push!(step_types, "FALLBACK")
    else
      @info "Bundle step successful (k=$(idx))"
      copyto!(z, bundle_step)
      push!(oracle_calls, bundle_calls)
      push!(step_types, "BUNDLE")
    end
    idx += 1
    push!(fvals, f(z))
    push!(elapsed_time, cumul_time)
    if (fvals[end] ≤ ϵ_tol) || (sum(oracle_calls) ≥ oracle_calls_limit)
      return SuperPolyak.SuperPolyakResult(
        z,
        fvals,
        oracle_calls,
        elapsed_time,
        step_types,
      )
    end
  end
end
