using LinearAlgebra
using Printf
using SparseArrays

import ReverseDiff: compile, gradient!, GradientTape
import QPSReader: readqps

using OSQP

import SuperPolyak

struct QuadraticProgram
  A::AbstractMatrix{Float64}
  P::AbstractMatrix{Float64}
  l::Vector{Float64}
  u::Vector{Float64}
  c::Vector{Float64}
end

"""
  read_problem_from_qps_file(filename::String, mpsformat; no_eqs::Bool = false) -> QuadraticProgram

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
  ℓ = problem.lcon
  u = problem.ucon
  A_aug = [A; sparse(1.0I, n, n)]
  return QuadraticProgram(
    A_aug,
    copy(P),
    [ℓ; problem.lvar],
    [u; problem.uvar],
    problem.c,
  )
end

function setup_osqp_model(qp::QuadraticProgram)
  model = OSQP.Model()
  OSQP.setup!(model, P=qp.P, q=qp.c, A=qp.A, l=qp.l, u=qp.u)
  return model
end

struct OsqpResult
  sol::Vector{Float64}
  iter::Int
  solve_time::Float64
  scale::Float64
  status_val::Int
end

"""
  solve_with_osqp(model::OSQP.Model,
                  primal_sol::Vector{Float64},
                  dual_sol::Vector{Float64};
                  ϵ_tol::Float64,
                  ϵ_rel::Float64,
                  iteration_limit::Int = 100000,
                  time_limit::Float64 = 0.0)

Solve a `QuadraticProgram` with OSQP up to tolerance `ϵ_tol`, starting from a
solution `(primal_sol, dual_sol)` within a given iteration and time limit.
A time limit of 0.0 indicates unlimited time.
"""
function solve_with_osqp(
  model::OSQP.Model,
  primal_sol::Vector{Float64},
  dual_sol::Vector{Float64};
  ϵ_tol::Float64,
  ϵ_rel::Float64,
  iteration_limit::Int = 100000,
  time_limit::Float64 = 0.0,
)
  # Update iteration and time limit.
  OSQP.update_settings!(
    model,
    max_iter = iteration_limit,
    time_limit = time_limit,
    eps_abs = ϵ_tol,
    eps_rel = ϵ_rel,
  )
  OSQP.warm_start!(model; x = primal_sol, y = dual_sol)
  result = OSQP.solve!(model)
  return OsqpResult(
    [result.x; result.y],
    result.info.iter,
    result.info.solve_time,
    result.info.rho_estimate,
    result.info.status_val,
  )
end

"""
  forward_backward_error(qp::QuadraticProgram)

Return a callable that computes the forward-backward error for a
`QuadraticProgram` solved using the OSQP splitting algorithm.
"""
function forward_backward_error(qp::QuadraticProgram)
  A = qp.A
  l = qp.l
  u = qp.u
  c = qp.c
  P = qp.P
  n = size(A, 2)
  # Expect length(z) = n + m.
  loss_fn(z::AbstractVector{Float64}) = begin
    x = z[1:n]
    y = z[(n+1):end]
    Ax = A * x
    diff_x = c + P*x + A'y      # x - (x - (c + Qx + A'y))
    diff_y = max.(l, min.(u, y + Ax)) - Ax
    return norm(diff_x) + norm(diff_y)
  end
  return loss_fn
end

fnorm(v::AbstractVector, v_norm::Float64) = begin
  return v_norm ≤ 1e-15 ? zero(v) : v / v_norm
end

"""
  forward_backward_error_subgradient(qp::QuadraticProgram)

Return a callable that computes the subgradient of the forward-backward error
for a `QuadraticProgram` solved using the OSQP splitting algorithm.
"""
function forward_backward_error_subgradient(qp::QuadraticProgram)
  A = qp.A
  l = qp.l
  u = qp.u
  c = qp.c
  P = qp.P
  m, n = size(A)
  g = zeros(m + n)
  grad_fn(z::AbstractVector{Float64}) = begin
    x = z[1:n]
    y = z[(n+1):end]
    Ax = A * x
    pri_res = c + P * x + A'y
    dua_res = max.(l, min.(u, y + Ax)) - Ax
    norm_pri = norm(pri_res)
    norm_dua = norm(dua_res)
    g[1:n] = P * fnorm(pri_res, norm_pri) +
      ((Diagonal(l .≤ (y + Ax) .≤ u) - I) * A)' * fnorm(dua_res, norm_dua)
    g[(n+1):end] = A * fnorm(pri_res, norm_pri) +
      (l .≤ (y + Ax) .≤ u) .* fnorm(dua_res, norm_dua)
    return g
  end
  return grad_fn
end

function subgradient_method(
  qp::QuadraticProgram,
  z₀::Vector{Float64},
  oracle_calls_limit::Int,
  ϵ_tol::Float64,
)
  loss = forward_backward_error(qp)
  grad = forward_backward_error_subgradient(qp)
  curr_iter = copy(z₀)
  for iter in 1:oracle_calls_limit
    f = loss(curr_iter)
    if f ≤ ϵ_tol
      return curr_iter, iter + 1
    end
    g = grad(curr_iter)
    curr_iter = curr_iter - f * g / (norm(g)^2)
    if (iter % 500 == 1)
      @info "subgradient_method - loss = $(loss(curr_iter)) - |g|: $(norm(g)) - tol: $(ϵ_tol)"
    end
  end
  return curr_iter, oracle_calls_limit
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
  model::OSQP.Model,
  x₀::Vector{Float64},
  y₀::Vector{Float64},
  exit_frequency::Int,
  oracle_calls_limit::Int,
  ϵ_tol::Float64,
  ϵ_rel::Float64,
)
  loss_fn = forward_backward_error(qp)
  m, n = size(qp.A)
  iter_total = 0
  time_total = 0.0
  z_prev = zeros(n + m)
  copyto!(z_prev, 1, x₀, 1, n)
  copyto!(z_prev, n+1, y₀, 1, m)
  while iter_total < oracle_calls_limit
    osqp_result = solve_with_osqp(
      model,
      z_prev[1:n],        # primal_sol
      z_prev[(n+1):end],  # dual_sol
      ϵ_tol = 1e-15,
      ϵ_rel = ϵ_rel,
      iteration_limit = exit_frequency,
    )
    iter_total += osqp_result.iter
    time_total += osqp_result.solve_time
    scale = osqp_result.scale
    if (osqp_result.status_val in [1; 2; -2])
      copyto!(z_prev, osqp_result.sol)
    else
      # Exit unsuccessfully if solver status was not 'Solved'.
      return OsqpResult(
        osqp_result.sol,
        iter_total,
        time_total,
        scale,
        osqp_result.status_val,
      )
    end
    new_loss = loss_fn(z_prev)
    @info "Fallback - it = $(iter_total) - loss = $(@sprintf("%.8f", new_loss))"
    # Exit successfully if forward-backward error was sufficiently reduced.
    if new_loss ≤ ϵ_tol
      return OsqpResult(z_prev, iter_total, time_total, scale, 1)
    end
  end
  return OsqpResult(z_prev, iter_total, time_total, 0.1, 2)
end

"""
  superpolyak_with_osqp(qp::QuadraticProgram, z₀::Vector{Float64};
                        ϵ_tol::Float64 = 1e-15, ϵ_rel::Float64 = 1e-15,
                        ϵ_decrease::Float64 = 1/2, ϵ_distance::Float64 = 3/2,
                        η_est::Float64 = 1.0, η_lb::Float64 = 0.1,
                        exit_frequency::Int = 250, oracle_calls_limit::Int = 100000,
                        bundle_budget::Int = length(z₀), budget_weight::Float64 = 0.0,
                        kwargs...)

Run SuperPolyak using the OSQP solver as the fallback method to solve a
`QuadraticProgram`. The loss used is the forward-backward error.
"""
function superpolyak_with_osqp(
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
  budget_weight::Float64 = 0.0,
  bundle_max_budget::Int = bundle_budget,
  bundle_step_threshold::Float64 = sqrt(ϵ_tol),
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
  # Create OSQP model
  osqp_model = setup_osqp_model(qp)
  z = z₀[:]
  fvals = [f(z₀)]
  oracle_calls = [0]
  elapsed_time = [0.0]
  step_types = ["NONE"]
  idx = 0
  # Number of variables and constraints.
  m, n = size(qp.A)
  @info "Using bundle system solver: QR_COMPACT_WV"
  # Initialize bundle budget.
  current_budget = bundle_budget
  while true
    cumul_time = 0.0
    Δ = fvals[end]
    η = ϵ_distance^(idx)
    target_tol = max(ϵ_decrease * Δ, ϵ_tol)
    # Remaining budget for the bundle method. If loss is not below threshold yet,
    # do not attempt a bundle step.
    budget_rem = (Δ ≤ bundle_step_threshold) ? min(current_budget, oracle_calls_limit - sum(oracle_calls)) : 0
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
      @info "Bundle step failed (k=$(idx), f=$(min(Δ, bundle_loss))) -- using fallback algorithm"
      fallback_result = fallback_algorithm(
        qp,
        osqp_model,
        z[1:n],
        z[(n+1):end],
        exit_frequency,
        # oracle_calls_limit
        max(oracle_calls_limit - (sum(oracle_calls) + bundle_calls), 1),
        # ϵ_tol
        target_tol,
        ϵ_rel,
      )
      copyto!(z, fallback_result.sol[1:end])
      fallback_calls = fallback_result.iter
      current_budget = min(
        Int(ceil(budget_weight * fallback_calls + (1 - budget_weight) * current_budget)),
        bundle_max_budget,
      )
      @info "Updating bundle budget: $(current_budget)"
      cumul_time += fallback_result.solve_time
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
