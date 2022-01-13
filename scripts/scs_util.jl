using LinearAlgebra
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
  num_equalities::Int
end

"""
  read_problem_from_qps_file(filename::String) -> OsqpProblem

Read a quadratic programming problem from a .QPS file and convert to the
representation used by the SCS solver.
"""
function read_problem_from_qps_file(filename::String, mpsformat)
  problem = readqps(filename, mpsformat=mpsformat)
  m, n = problem.ncon, problem.nvar
  # The objective matrix is symmetric and the .QPS file gives
  # the lower-triangular part only.
  P = sparse(problem.qrows, problem.qcols, problem.qvals, n, n)
  P = P + tril(P, 1)'
  A = sparse(problem.arows, problem.acols, problem.avals, m, n)
  # Constraints on variables should be included in the constraint matrix.
  finite_ind = @. (1:n)[!isinf(problem.lvar)|!isinf(problem.uvar)]
  @assert norm(problem.ucon - problem.lcon, Inf) ≤ 1e-15 "Expected equality constraints only"
  # Augmented matrix in the system s - Ax in K.
  A_var = [-sparse(1.0I, n, n); sparse(1.0I, n, n)]
  b_var = [-problem.lvar; problem.uvar]
  # Only include nontrivial variable bounds (SCS will yield nan's otherwise).
  finite_ind = isfinite.(b_var)
  A_aug = [A; A_var[finite_ind, :]]
  b_aug = [problem.ucon; b_var[finite_ind]]
  return QuadraticProgram(
    A_aug,
    copy(P),
    b_aug,
    problem.c,
    m,
  )
end

struct ScsResult
  sol::Vector{Float64}
  iter::Int
  solve_time::Float64
  scale::Float64
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
    qp.num_equalities,
    m - qp.num_equalities,
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
    scale = initial_scale,
    max_iters = max(iteration_limit, 1),
  )
  # Convert time elapsed from milliseconds to seconds.
  return ScsResult(
    [result.x; result.y; result.s],
    result.info.iter,
    1e-3 * result.info.solve_time,
    result.info.scale,
  )
end

"""
  proj_cone!(qp::QuadraticProgram, y::Vector{Float64}, s::Vector{Float64})

Project `(s, y)` to the cone `K × K*` specified by a `QuadraticProgram`.
"""
function proj_cone!(
  qp::QuadraticProgram,
  y::Vector{Float64},
  s::Vector{Float64},
)
  num_eq = qp.num_equalities
  # s should be 0 on 1:(num_eq)
  s[1:num_eq] .= 0.0
  # s and y should be nonnegative on (num_eq+1):m and
  # s_i y_i should be 0 everywhere.
  # If s_i and y_i > 0, we set the smallest one to 0
  # to project to set (s, y): s_i y_i = 0.
  # NOTE: both_positive is contained in [(num_eq+1):m].
  both_positive = (s .> 0) .& (y .> 0)
  s[s .< 0] .= 0.0
  y[y .< 0] .= 0.0
  s_minimizer = (s .< y) .& both_positive
  y_minimizer = (y .< s) .& both_positive
  s[s_minimizer] .= 0.0
  y[y_minimizer] .= 0.0
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
  # Range of linear inequalities.
  ineq_range = (qp.num_equalities+1):m
  loss_fn(z::AbstractVector) = begin
    x = z[1:n]
    y = z[(n+1):(n+m)]
    s = z[(n+m+1):end]
    Px = P * x
    return max(
      norm(A * x + s - b, Inf),
      norm(Px + A'y + c, Inf),
      abs(x' * Px + c'x + b'y),
      # s in K
      norm(min.(s[ineq_range], 0.0), Inf),
      norm(s[1:qp.num_equalities], Inf),
      # y in K∗
      norm(min.(y[ineq_range], 0.0), Inf),
      # s \perp y
      norm(s .* y, Inf),
    )
  end
  return loss_fn
end

function kkt_error_subgradient(qp::QuadraticProgram)
  m, n = size(qp.A)
  compiled_loss_tape = compile(GradientTape(kkt_error(qp), randn(2m + n)))
  return z -> gradient!(compiled_loss_tape, z)
end

function superpolyak_with_scs(
  qp::QuadraticProgram,
  z₀::Vector{Float64};
  ϵ_tol::Float64 = 1e-15,
  ϵ_rel::Float64 = 1e-15,
  ϵ_decrease::Float64 = (1 / 2),
  ϵ_distance::Float64 = (3 / 2),
  η_est::Float64 = 1.0,
  η_lb::Float64 = 0.1,
  oracle_calls_limit::Int = 100000,
  bundle_budget::Int = length(z₀),
  kwargs...,
)
  f = kkt_error(qp)
  g = kkt_error_subgradient(qp)
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
  x = z₀[:]
  fvals = [f(z₀)]
  oracle_calls = [0]
  elapsed_time = [0.0]
  step_types = ["NONE"]
  idx = 0
  # Initial dual scale factor for calls to SCS.
  scs_scale = 0.1
  # Number of variables and constraints.
  m, n = size(qp.A)
  @info "Using bundle system solver: QR_COMPACT_WV"
  while true
    cumul_time = 0.0
    Δ = fvals[end]
    η = ϵ_distance^(idx)
    target_tol = max(ϵ_decrease * Δ, ϵ_tol)
    # Remaining budget for the bundle method.
    budget_rem = min(bundle_budget, oracle_calls_limit - sum(oracle_calls))
    bundle_stats = @timed bundle_step, bundle_calls =
      SuperPolyak.build_bundle_wv(f, g, x, η, 0.0, η_est, budget_rem)
    cumul_time += bundle_stats.time - bundle_stats.gctime
    # Adjust η_est if the bundle step did not satisfy the descent condition.
    bundle_loss = isnothing(bundle_step) ? Δ : f(bundle_step)
    if !isnothing(bundle_step) && (bundle_loss > Δ^(1 + η_est)) && (Δ < 0.5)
      η_est = max(η_est * 0.9, η_lb)
      @debug "Adjusting η_est = $(η_est)"
    end
    if isnothing(bundle_step) || (bundle_loss > target_tol)
      if (!isnothing(bundle_step)) && (bundle_loss < Δ)
        copyto!(x, bundle_step)
      end
      @info "Bundle step failed (k=$(idx), f=$(bundle_loss)) -- using fallback algorithm"
      fallback_stats = @timed scs_result =
        solve_with_scs(
          qp,
          x[1:n],
          x[(n+1):(n+m)],
          x[(n+m+1):end],
          ϵ_tol = target_tol,
          ϵ_rel = ϵ_rel,
          initial_scale = scs_scale,
          iteration_limit = max(
            oracle_calls_limit - (sum(oracle_calls) + bundle_calls),
            0,
          ),
        )
      copyto!(x, scs_result.sol)
      fallback_calls = scs_result.iter
      @info "Updating scs scale: $(scs_result.scale) from $(scs_scale)"
      scs_scale = scs_result.scale
      cumul_time += fallback_stats.time - fallback_stats.gctime
      # Include the number of oracle calls made by the failed bundle step.
      push!(oracle_calls, fallback_calls + bundle_calls)
      push!(step_types, "FALLBACK")
    else
      @info "Bundle step successful (k=$(idx))"
      x = bundle_step[:]
      push!(oracle_calls, bundle_calls)
      push!(step_types, "BUNDLE")
    end
    idx += 1
    push!(fvals, f(x))
    push!(elapsed_time, cumul_time)
    if (fvals[end] ≤ ϵ_tol) || (sum(oracle_calls) ≥ oracle_calls_limit)
      return SuperPolyak.SuperPolyakResult(
        x,
        fvals,
        oracle_calls,
        elapsed_time,
        step_types,
      )
    end
  end
end
