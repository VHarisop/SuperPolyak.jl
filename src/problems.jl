"""
  problems.jl: Implementations of standard problems from machine learning and
  signal processing for the bundle Newton framework.
"""

struct PhaseRetrievalProblem
  A::AbstractMatrix
  x::AbstractVector
  y::AbstractVector
end

"""
  loss(problem::PhaseRetrievalProblem)

Return a callable that computes the robust ℓ₁ loss for a phase retrieval
problem.
"""
function loss(problem::PhaseRetrievalProblem)
  return z ->
    (1 / length(problem.y)) * norm(abs.(problem.A * z) .- problem.y, 1)
end

"""
  loss(problem::PhaseRetrievalProblem)

Return a callable that computes a subgradient of the robust ℓ₁ loss for a phase
retrieval problem.
"""
function subgradient(problem::PhaseRetrievalProblem)
  d = size(problem.A, 2)
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(d)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  initializer(problem::PhaseRetrievalProblem, δ::Float64)

Return an initial estimate `x₀` that is `δ`-close to the ground truth in
normalized distance.
"""
function initializer(problem::PhaseRetrievalProblem, δ::Float64)
  return problem.x + δ * normalize(randn(eltype(problem.x), length(problem.x)))
end

"""
  phase_retrieval_problem(m, d; elt_type = Float64)

Generate a phase retrieval problem in dimension `d` with `m` random Gaussian
measurements.
"""
function phase_retrieval_problem(m, d; elt_type = Float64)
  A = randn(elt_type, (m, d))
  x = normalize(randn(elt_type, d))
  return PhaseRetrievalProblem(A, x, abs.(A * x))
end

# Phase, defined for both complex and real vectors.
phase(v::AbstractVector) = begin
  norm_fn(z::Number) = abs(z) ≤ 1e-15 ? 1.0 : z / abs(z)
  return norm_fn.(v)
end

fnorm(v::AbstractVector, nrm::Float64) = nrm ≤ 1e-15 ? zero(v) : (v / nrm)

"""
  loss_altproj(problem::PhaseRetrievalProblem)

Return a callable that computes the loss of the alternating projections method
for phase retrieval.
"""
function loss_altproj(problem::PhaseRetrievalProblem)
  A = problem.A
  y = problem.y
  m = length(y)
  # Cache factorization of A.
  F = qr(A)
  loss_fn(z::AbstractVector) = begin
    z_comp = z[1:m] + z[(m+1):end] * im
    return norm(z_comp - A * (F \ z_comp)) + norm(z_comp - y .* phase(z_comp))
  end
end

"""
  subgradient_altproj(problem::PhaseRetrievalProblem)

Return a callable that computes a subgradient of the loss of the alternating
projections method for phase retrieval.
"""
function subgradient_altproj(problem::PhaseRetrievalProblem)
  A = problem.A
  y = problem.y
  m = length(y)
  # Cache factorization of A.
  F = qr(A)
  # Cache subgradients.
  g = zeros(2 * m)
  g_comp = zeros(eltype(A), m)
  grad_fn(z::AbstractVector) = begin
    z_comp = z[1:m] + z[(m+1):end] * im
    diff_range = z_comp - A * (F \ z_comp)
    diff_phase = z_comp - y .* phase(z_comp)
    norm_range = norm(diff_range)
    norm_phase = norm(diff_phase)
    # Separate real and imaginary parts in the subgradient.
    copyto!(
      g_comp,
      fnorm(diff_range, norm_range) + fnorm(diff_phase, norm_phase),
    )
    g[1:m] = real.(g_comp)
    g[(m+1):end] = imag.(g_comp)
    return g
  end
  return grad_fn
end

"""
  alternating_projections_step(problem::PhaseRetrievalProblem)

Return a callable that computes an alternating projections step for a phase
retrieval problem.
"""
function alternating_projections_step(problem::PhaseRetrievalProblem)
  A = problem.A
  y = problem.y
  m = length(y)
  F = qr(A)
  step_fn(z::AbstractVector) = begin
    phased = phase(A * (F \ (z[1:m] + z[(m+1):end] * im)))
    return [
      y .* real.(phased);
      y .* imag.(phased)
    ]
  end
  return step_fn
end

"""
  initializer_altproj(problem::PhaseRetrievalProblem, δ::Float64)

Return an initial guess for the alternating projections algorithm for phase
retrieval generated by a point `δ`-close to the ground truth.
"""
function initializer_altproj(problem::PhaseRetrievalProblem, δ::Float64)
  x = problem.x + δ * normalize(randn(eltype(problem.x), length(problem.x)))
  Ax = problem.A * x
  return [real.(Ax); imag.(Ax)]
end

"""
  QuadraticSensingProblem

A symmetrized quadratic sensing problem with measurements

  yᵢ = |pᵢ'X|² - |qᵢ'X|² = (pᵢ - qᵢ)'XX'(pᵢ + qᵢ)

where pᵢ and qᵢ are iid Gaussian vectors.

We let ℓᵢ = pᵢ - qᵢ, rᵢ = pᵢ + qᵢ below.
"""
struct QuadraticSensingProblem
  L::Matrix{Float64}
  R::Matrix{Float64}
  X::Matrix{Float64}
  y::Vector{Float64}
end

function loss(problem::QuadraticSensingProblem)
  d, k = size(problem.X)
  m = length(problem.y)
  L = problem.L
  R = problem.R
  y = problem.y
  loss_fn(z) = begin
    Z = reshape(z, d, k)
    return (1 / m) * norm(y - sum((L * Z) .* (R * Z), dims = 2)[:], 1)
  end
  return loss_fn
end

function subgradient(problem::QuadraticSensingProblem)
  d, r = size(problem.X)
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(d * r)))
  return z -> gradient!(compiled_loss_tape, z)
end

function initializer(problem::QuadraticSensingProblem, δ::Float64)
  Δ = randn(size(problem.X))
  return problem.X + δ * (Δ / norm(Δ))
end

function quadratic_sensing_problem(m::Int, d::Int, r::Int)
  X = Matrix(qr(randn(d, r)).Q)
  L = sqrt(2) * randn(m, d)
  R = sqrt(2) * randn(m, d)
  y = sum((L * X) .* (R * X), dims = 2)[:]
  return QuadraticSensingProblem(L, R, X, y)
end

"""
  BilinearSensingProblem

A bilinear sensing problem with measurements

  `yᵢ = ℓᵢ'W * X'rᵢ`

where `W` and `X` are `d × r` matrices.
"""
struct BilinearSensingProblem
  L::Matrix{Float64}
  R::Matrix{Float64}
  W::Matrix{Float64}
  X::Matrix{Float64}
  y::Vector{Float64}
end

"""
  loss(problem::BilinearSensingProblem)

Implement the ℓ₁ robust loss for a bilinear sensing problem with general rank.
Assumes that the argument will be a vector containing the "flattened" version
of the matrix `[W, X]`, where `W` and `X` are `d × r` matrices.
"""
function loss(problem::BilinearSensingProblem)
  L = problem.L
  R = problem.R
  y = problem.y
  m = length(y)
  d, r = size(problem.W)
  return z -> begin
    # Layout assumption: z is a "flattened" version of [W, X] ∈ Rᵈˣ⁽²ʳ⁾.
    W = reshape(z[1:(d*r)], d, r)
    X = reshape(z[(d*r+1):end], d, r)
    # Compute row-wise product.
    return (1 / m) * norm(y .- sum((L * W) .* (R * X), dims = 2)[:], 1)
  end
end

"""
  subgradient(problem::BilinearSensingProblem)

Implement the subgradient of the ℓ₁ robust loss for a bilinear sensing problem
with general rank. Like `loss(problem)`, assumes that the argument will be a
vector containing the "flattened" version of the matrix `[W, X]`, where `W` and
`X` are `d × r` matrices.
"""
function subgradient(problem::BilinearSensingProblem)
  d, r = size(problem.W)
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(2 * d * r)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  bilinear_sensing_problem(m::Int, d::Int, r::Int)

Generate a bilinear sensing problem with solutions of dimension `d × r` and `m`
measurements using random Gaussian sensing matrices.
"""
function bilinear_sensing_problem(m::Int, d::Int, r::Int)
  L = randn(m, d)
  R = randn(m, d)
  # Solutions on the orthogonal manifold O(d, r).
  W = Matrix(qr(randn(d, r)).Q)
  X = Matrix(qr(randn(d, r)).Q)
  y = sum((L * W) .* (R * X), dims = 2)[:]
  return BilinearSensingProblem(L, R, W, X, y)
end

"""
  initializer(problem::BilinearSensingProblem, δ::Float64)

Generate an initial guess for the solution to `problem` that is `δ`-far from
the ground truth when distance is measured in the Euclidean norm.
"""
function initializer(problem::BilinearSensingProblem, δ::Float64)
  wx_stacked = [vec(problem.W); vec(problem.X)]
  return wx_stacked + δ * normalize(randn(length(wx_stacked)))
end

"""
  generate_sparse_vector(d, k)

Generate a random length-`d` vector of unit norm with `k` nonzero elements.
"""
function generate_sparse_vector(d, k)
  x = zeros(d)
  x[sample(1:d, k, replace = false)] = normalize(randn(k))
  return x
end

struct ReluRegressionProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
end

"""
  loss(problem::ReluRegressionProblem)

Return a callable that computes the ℓ₁ robust loss for a ReLU regression
problem.
"""
function loss(problem::ReluRegressionProblem)
  A = problem.A
  y = problem.y
  m = length(y)
  return z -> (1 / m) * norm(max.(A * z, 0.0) - y, 1)
end

"""
  subgradient(problem::ReluRegressionProblem)

Return a callable that computes a subgradient of the ℓ₁ robust loss for a ReLU
regression problem.
"""
function subgradient(problem::ReluRegressionProblem)
  d = size(problem.A, 2)
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(d)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  initializer(problem::ReluRegressionProblem, δ::Float64)

Return an initial estimate of the unknown vector in the ReLU regression
`problem` with normalized distance `δ` from the ground truth.
"""
function initializer(problem::ReluRegressionProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end

"""
  relu_regression_problem(m, d)

Create a random instance of ReLU regression problem with random Gaussian
measurement vectors.
"""
function relu_regression_problem(m, d)
  A = randn(m, d)
  x = normalize(randn(d))
  return ReluRegressionProblem(A, x, max.(A * x, 0.0))
end

struct MaxAffineRegressionProblem
  A::Matrix{Float64}
  # True slopes of each affine piece, one per column.
  βs::Matrix{Float64}
  y::Vector{Float64}
end

"""
  loss(problem::MaxAffineRegressionProblem)

Return a callable that computes the ℓ₁ robust loss for a max-linear regression
problem.
"""
function loss(problem::MaxAffineRegressionProblem)
  A = problem.A
  y = problem.y
  m = length(y)
  d, k = size(problem.βs)
  # Assumes the input is a flattened version of `βs`, so it must be reshaped
  # before applying the operator `A`.
  return z -> (1 / m) * norm(maximum(A * reshape(z, d, k), dims = 2)[:] - y, 1)
end

"""
  subgradient(problem::MaxAffineRegressionProblem)

Return a callable that computes a subgradient of the ℓ₁ robust loss for a
max-linear regression problem.
"""
function subgradient(problem::MaxAffineRegressionProblem)
  d, k = size(problem.βs)
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(d * k)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  initializer(problem::MaxAffineRegressionProblem, δ::Float64)

Return an initial estimate of the slopes of each affine piece for a max-affine
regression `problem` with normalized distance `δ` from the ground truth.
"""
function initializer(problem::MaxAffineRegressionProblem, δ::Float64)
  βs = problem.βs[:]
  return βs + (δ * norm(βs)) .* normalize(randn(length(βs)))
end

"""
  max_affine_regression_problem(m, d, k)

Create a random instance of a max-linear regression problem with random
Gaussian measurements.
"""
function max_affine_regression_problem(m, d, k)
  A = randn(m, d)
  βs = mapslices(normalize, randn(d, k), dims = 1)
  return MaxAffineRegressionProblem(A, βs, maximum(A * βs, dims = 2)[:])
end

struct CompressedSensingProblem
  A::Matrix{Float64}
  # Store a factorization to compute projections quickly.
  F::Factorization{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
  k::Int
end

"""
  proj_sparse(x::Vector{Float64}, k::Int)

Project the vector `x` to the cone of `k`-sparse vectors. Modifies
the original vector `x`.
"""
function proj_sparse(x::Vector{Float64}, k::Int)
  x[sortperm(abs.(x), rev = true)[(k+1):end]] .= 0
  return x
end

function dist_sparse(x::Vector{Float64}, k::Int)
  return norm(x - proj_sparse(x[:], k))
end

function grad_sparse(x::Vector{Float64}, k::Int)
  x₊ = proj_sparse(x[:], k)
  ds = norm(x₊ - x)
  return (ds ≤ 1e-15) ? zeros(length(x)) : (x - x₊) / ds
end

"""
  proj_range(problem::CompressedSensingProblem, x::Vector{Float64})

For a compressed sensing problem of the form `y = Ax`, where `A` is a
short matrix, computes the projection of `x` onto the range of `A`.
"""
function proj_range(problem::CompressedSensingProblem, x::Vector{Float64})
  return x + problem.F \ (problem.y - problem.A * x)
end

function dist_range(problem::CompressedSensingProblem, x::Vector{Float64})
  return norm(problem.F \ (problem.y - problem.A * x))
end

function grad_range(problem::CompressedSensingProblem, x::Vector{Float64})
  dx = problem.F \ (problem.y - problem.A * x)
  ds = norm(dx)
  return (ds ≤ 1e-15) ? zeros(length(x)) : -dx / ds
end

function loss(problem::CompressedSensingProblem)
  return z -> dist_sparse(z, problem.k) + dist_range(problem, z)
end

function subgradient(problem::CompressedSensingProblem)
  return z -> grad_sparse(z, problem.k) + grad_range(problem, z)
end

function compressed_sensing_problem(m, d, k)
  A = randn(m, d)
  # Store the pivoted QR factorization for A.
  F = qr(A, Val(true))
  x = generate_sparse_vector(d, k)
  return CompressedSensingProblem(A, F, x, A * x, k)
end

"""
  initializer(problem::CompressedSensingProblem, δ::Float64)

Return an initial estimate of the solution of a compressed sensing `problem`
with normalized distance `δ` from the ground truth.
"""
function initializer(problem::CompressedSensingProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end
