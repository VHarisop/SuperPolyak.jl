using LinearAlgebra
using SparseArrays

import ReverseDiff: compile, gradient!, GradientTape
import QPSReader: readqps
import SCS: IndirectSolver, DirectSolver, scs_solve

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
  l_var = problem.lvar[finite_ind]
  u_var = problem.uvar[finite_ind]
  @assert norm(problem.ucon - problem.lcon, Inf) ≤ 1e-15 "Expected equality constraints only"
  return QuadraticProgram(
    [A; -I; I],
    P,
    [problem.ucon; -l_var; u_var],
    problem.c,
    problem.ncon,
  )
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
  slack::Vector{Float64},
  ϵ_tol::Float64,
  use_direct_solver::Bool = true,
)
  m, n = size(qp.A)
  return scs_solve(
    use_direct_solver ? DirectSolver : IndirectSolver,
    m,
    n,
    qp.A,
    qp.P,
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
    eps_rel = 1e-15,
  )
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
      norm(A * x + s - b, Inf),
      norm(Px + A'y - c, Inf),
      abs(x' * Px + c'x + b'y),
    )
  end
  return loss_fn
end

function kkt_error_subgradient_autodiff(qp::QuadraticProgram)
  m, n = size(qp.A)
  compiled_loss_tape = compile(GradientTape(kkt_error(qp), randn(2m + n)))
  return z -> gradient!(compiled_loss_tape, z)
end

fnorm(v::AbstractVector, nrm_v::Float64) = nrm_v ≤ 1e-15 ? zero(v) : v / nrm_v

"""
  kkt_error_subgradient(qp::QuadraticProgram)

Return a callable that computes a subgradient of the KKT error of a
`QuadraticProgram`.
"""
function kkt_error_subgradient(qp::QuadraticProgram)
  A = qp.A
  b = qp.b
  c = qp.c
  P = qp.P
  m, n = size(A)
  # Preallocate x, y, s
  x = zeros(n)
  y = zeros(m)
  s = zeros(m)
  # Preallocate subgradient
  g = zeros(2m + n)
  grad_fn(z::AbstractVector) = begin
    fill!(g, 0.0)
    copyto!(x, 1, z, 1, n)
    copyto!(y, 1, z, n+1, m)
    copyto!(s, 1, z, n+m+1, m)
    Px = P * x
    Ax = A * x
    pri_res = Ax + s - b
    dua_res = Px + A'y - c
    gap_res = x'Px + c'x + b'y
    # Find which term maximizes the kkt error.
    pri_norm = norm(pri_res)
    dua_norm = norm(dua_res)
    gap = abs(gap_res)
    loss = maximum(pri_norm, dua_norm, gap)
    if pri_norm == loss
      g[1:n] = A' * fnorm(pri_res, pri_norm)
      g[(n+m+1):end] = fnorm(pri_res, pri_norm)
    elseif dua_norm == loss
      g[1:n] = P * fnorm(dua_res, dua_norm)
      g[(n+1):(n+m)] = A * fnorm(dua_res, dua_norm)
    else
      g[1:n] = sign(gap) * (Px + c)
      g[(n+1):(n+m)] = sign(gap) * b
    end
    return g
  end
  return grad_fn
end
