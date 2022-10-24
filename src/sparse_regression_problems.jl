function soft_threshold(x::Vector{Float64}, τ::Float64)
  return sign.(x) .* max.(abs.(x) .- τ, 0.0)
end

struct LassoProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
  λ::Float64
end

"""
  proximal_gradient(problem::LassoProblem, x::Vector{Float64}, τ::Float64)

Compute the proximal operator for the Lasso problem

  min_x |Ax - b|^2 + λ |x|₁,

with step size `τ`.
"""
function proximal_gradient(
  problem::LassoProblem,
  x::Vector{Float64},
  τ::Float64,
)
  return soft_threshold(
    x - τ * problem.A' * (problem.A * x - problem.y),
    problem.λ * τ,
  )
end

"""
  loss(problem::LassoProblem, τ::Float64 = 0.95 / (opnorm(problem.A)^2))

Compute the residual of the proximal gradient method applied to solve a LASSO
regression `problem` with step `τ`.
"""
function loss(problem::LassoProblem, τ::Float64 = 0.95 / (opnorm(problem.A)^2))
  A = problem.A
  y = problem.y
  λ = problem.λ
  loss_fn(z) = begin
    grad_step = z - τ * A' * (A * z - y)
    return norm(z - sign.(grad_step) .* max.(abs.(grad_step) .- λ * τ, 0.0))
  end
  return loss_fn
end

function subgradient(
  problem::LassoProblem,
  τ::Float64 = 0.95 / (opnorm(problem.A)^2),
)
  return z -> gradient(loss(problem, τ), z)[1]
end

"""
  support_recovery_lambda(A::Matrix{Float64}, x::Vector{Float64},
                          σ::Float64)

Compute a value for `λ` that guarantees recovery of a support contained within
the support of the ground truth solution under normalized Gaussian designs.
"""
function support_recovery_lambda(
  A::Matrix{Float64},
  x::Vector{Float64},
  σ::Float64,
)
  m, d = size(A)
  nnz_ind = abs.(x) .> 1e-15
  # Compute factor γ = 1 - |X_{S^c}'X_S (X_S'X_S)^{-1}|_{∞}.
  S = A[:, nnz_ind]
  T = A[:, .!nnz_ind]
  γ = 1.0 - opnorm(T' * (S * inv(S'S)), Inf)
  @info "γ = $(γ)"
  return (2.0 / γ) * sqrt(σ^2 * log(d) / m)
end

function lasso_problem(m::Int, d::Int, k::Int, σ::Float64 = 0.1; kwargs...)
  x = generate_sparse_vector(d, k)
  A = Matrix(qr(randn(d, m)).Q)'
  y = A * x + σ .* randn(m)
  return LassoProblem(A, x, y, get(kwargs, :λ, 0.2 * norm(A'y, Inf)))
end

function compute_tau(problem::LassoProblem)
  return 0.95 / (opnorm(problem.A)^2)
end

function initializer(problem::LassoProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end

"""
  GenerativeSensingProblem

A problem with measurements

  yᵢ = A ⋅ G(x),

where A is a random Gaussian matrix and G is a ReLU network with user-provided
hidden layer sizes.
"""
struct GenerativeSensingProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
  # Matrices of each hidden layer.
  Ws::Vector{Matrix{Float64}}
end

"""
  generative_sensing_problem(m::Int, k::Int, hidden_layers::Vector{Int})

Create a generative sensing problem with `m` measurements, latent signal of
dimension `k`, and a number of hidden layer sizes given by `hidden_layers`.
"""
function generative_sensing_problem(m::Int, k::Int, hidden_layers::Vector{Int})
  Ws = [(1 / sqrt(hidden_layers[i])) * randn(hidden_layers[i], hidden_layers[i+1])
        for i in 1:(length(hidden_layers) - 1)]
  push!(Ws, (1 / sqrt(hidden_layers[end])) * randn(hidden_layers[end], k))
  x = normalize(randn(k))
  A = randn(m, hidden_layers[1])
  y = x[:]
  for W in reverse(Ws)
    y = max.(W * y, 0.0)
  end
  return GenerativeSensingProblem(A, x, A * y, Ws)
end

function generative_model_output(
  problem::GenerativeSensingProblem,
  z::AbstractVector{Float64},
)
  y_temp = z[:]
  for W in reverse(problem.Ws)
    y_temp = max.(W * y_temp, 0.0)
  end
  return y_temp
end

function loss(problem::GenerativeSensingProblem)
  A = problem.A
  y = problem.y
  return z -> (1 / length(y)) * norm(A * generative_model_output(problem, z) .- y, 1)
end

function subgradient(problem::GenerativeSensingProblem)
  return z -> gradient(loss(problem), z)[1]
end

function initializer(problem::GenerativeSensingProblem, δ::Float64)
  return problem.x + δ * normalize(randn(size(problem.x)))
end
