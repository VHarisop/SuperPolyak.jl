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
  problem = readqps(filename, mpsformat = mpsformat)
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
  OSQP.setup!(model, P = qp.P, q = qp.c, A = qp.A, l = qp.l, u = qp.u)
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
  time_limit::Float64 = 0.0
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
    diff_x = c + P * x + A'y      # x - (x - (c + Qx + A'y))
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

"""
  forward_backward_error_oracle(qp::QuadraticProgram)

Return a first-order oracle for the forward-backward error of a `QuadraticProgram`
solved using the OSQP splitting algorithm.
"""
function forward_backward_error_oracle(qp::QuadraticProgram)
  A = qp.A
  l = qp.l
  u = qp.u
  c = qp.c
  P = qp.P
  m, n = size(A)
  # Preallocate all vectors
  x = zeros(n)
  y = zeros(m)
  pri_res = zeros(n)
  dua_res = zeros(m)
  g = zeros(n + m)
  Ax = zeros(m)
  bv = BitVector(undef, m)
  oracle_fn(z::AbstractVector{Float64}) = begin
    copyto!(x, 1, z, 1, n)
    copyto!(y, 1, z, n + 1, m)
    copyto!(Ax, A * x)
    pri_res .= c .+ P * x .+ A'y
    dua_res .= max.(l, min.(u, y .+ Ax)) .- Ax
    nrm_pri = norm(pri_res)
    nrm_dua = norm(dua_res)
    if nrm_pri .> 1e-15
      pri_res ./= nrm_pri
    end
    if nrm_dua .> 1e-15
      dua_res ./= nrm_dua
    end
    # Bit-vector containing the active indices.
    bv .= l .≤ (y + Ax) .≤ u
    g[1:n] .= P * pri_res .+ A' * ((Int.(bv) .- 1) .* dua_res)
    g[(n+1):end] .= A * pri_res .+ (bv .* dua_res)
    return (nrm_pri + nrm_dua, g)
  end
  return oracle_fn
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
the loss function. It uses the OSQP solver to find a solution `(x, y)` with
(absolute) error at most `ϵ_tol`.
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
  copyto!(z_prev, n + 1, y₀, 1, m)
  while iter_total < oracle_calls_limit
    osqp_result = solve_with_osqp(
      model,
      z_prev[1:n],        # primal_sol
      z_prev[(n+1):end],  # dual_sol
      ϵ_tol = 1e-15,
      ϵ_rel = ϵ_rel,
      iteration_limit = min(oracle_calls_limit - iter_total, exit_frequency),
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

function update_budget(budget::Int, oracle_calls::Int, weight::Float64, max_budget::Int)
  return min(
    Int(ceil(budget * (1 - weight) + weight * oracle_calls)),
    max_budget,
  )
end

"""
  build_bundle_wv(first_order_oracle::Function, y₀::Vector{Float64}, η::Float64,
                  min_f::Float64, η_est::Float64, bundle_budget::Int = length(y₀))

An efficient version of the BuildBundle algorithm using an incrementally updated
QR algorithm based on the compact WV representation.
"""
function build_bundle_wv(
  first_order_oracle::Function,
  y₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
  η_est::Float64,
  bundle_budget::Int = length(y₀),
)
  d = length(y₀)
  bundle_budget = min(bundle_budget, d)
  if bundle_budget ≤ 0
    return y₀, 0
  end
  bvect = zeros(d)
  fvals = zeros(bundle_budget)
  y = y₀[:]
  # To obtain a solution equivalent to applying the pseudoinverse, we use the
  # QR factorization of the transpose of the bundle matrix. This is because the
  # minimum-norm solution to Ax = b when A is full row rank can be found via the
  # QR factorization of Aᵀ.
  f₀, g₀ = first_order_oracle(y)
  copyto!(bvect, g₀)
  fvals[1] = f₀ - min_f + bvect' * (y₀ - y)
  # Initialize Q and R
  Q, R = SuperPolyak.wv_from_vector(bvect)
  y = y₀ - bvect' \ fvals[1]
  f, g = first_order_oracle(y)
  resid = f - min_f
  copyto!(bvect, g)
  Δ = f₀ - min_f
  # Exit early if solution escaped ball.
  if norm(y - y₀) > η * Δ
    return nothing, 1
  end
  # Best solution and function value found so far.
  y_best = y[:]
  f_best = resid[1]
  # Cache right-hand side vector.
  qr_rhs = zero(y)
  for bundle_idx in 2:bundle_budget
    # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
    fvals[bundle_idx] = resid + bvect' * (y₀ - y)
    # Update the QR decomposition of A' after forming [A' bvect].
    # Q is updated in-place.
    SuperPolyak.qrinsert_wv!(Q, R, bvect)
    # Terminate early if rank-deficient.
    # size(R) = (d, bundle_idx).
    if R[bundle_idx, bundle_idx] < 1e-15
      @debug "Stopping (idx=$(bundle_idx)) - reason: rank-deficient A"
      y = y₀ - Matrix(Q * R)' \ view(fvals, 1:bundle_idx)
      return (norm(y - y₀) ≤ η * Δ ? y : y_best), bundle_idx
    end
    # Update y by solving the system Q * (inv(R)'fvals).
    # Cost: O(d * bundle_idx)
    Rupper = view(R, 1:bundle_idx, 1:bundle_idx)
    qr_rhs[1:bundle_idx] = Rupper' \ fvals[1:bundle_idx]
    y = y₀ - Q * qr_rhs
    f, g = first_order_oracle(y)
    resid = f - min_f
    copyto!(bvect, g)
    # Terminate early if new point escaped ball around y₀.
    if (norm(y - y₀) > η * Δ)
      @debug "Stopping at idx = $(bundle_idx) - reason: diverging"
      return y_best, bundle_idx
    end
    # Terminate early if function value decreased significantly.
    if (Δ < 0.5) && (resid < Δ^(1 + η_est))
      return y, bundle_idx
    end
    # Otherwise, update best solution so far.
    if (resid < f_best)
      copyto!(y_best, y)
      f_best = resid
    end
  end
  return y_best, bundle_budget
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
  kwargs...
)
  first_order_oracle = forward_backward_error_oracle(qp)
  f = forward_backward_error(qp)
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
  n = size(qp.A, 2)
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
      build_bundle_wv(first_order_oracle, z, η, 0.0, η_est, budget_rem)
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
      current_budget = update_budget(
        current_budget,
        fallback_calls,
        budget_weight,
        bundle_max_budget,
      )
      @info "Updating bundle budget: $(current_budget)"
      cumul_time += fallback_result.solve_time
      # Include the number of oracle calls made by the failed bundle step.
      push!(oracle_calls, fallback_calls + bundle_calls)
      push!(step_types, "FALLBACK")
    else
      @info "Bundle step successful (k=$(idx), f=$(bundle_loss))"
      copyto!(z, bundle_step)
      current_budget = update_budget(
        current_budget,
        bundle_calls,
        budget_weight,
        bundle_max_budget,
      )
      @info "Updating bundle budget: $(current_budget)"
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
