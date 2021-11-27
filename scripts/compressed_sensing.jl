using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

function run_experiment(m, d, k, δ, ϵ_decrease, ϵ_distance, show_amortized)
  problem = SuperPolyak.compressed_sensing_problem(m, d, k)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init = SuperPolyak.initializer(problem, δ)
  _, loss_history, oracle_calls = SuperPolyak.bundle_newton(
    loss_fn,
    grad_fn,
    x_init,
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
  )
  T = length(oracle_calls)
  cumul_oracle_calls =
    show_amortized ? ((1:T) .* (sum(oracle_calls) ÷ T)) : cumsum(oracle_calls)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("compressed_sensing_$(m)_$(d)_$(k).csv", df_bundle)
  semilogy(cumul_oracle_calls, loss_history, "bo--")
  show()
end

settings = ArgParseSettings()
@add_arg_table! settings begin
  "--m"
    arg_type = Int
    help = "The number of measurements."
    default = 50
  "--d"
    arg_type = Int
    help = "The problem dimension."
    default = 500
  "--k"
    arg_type = Int
    help = "The sparsity of the unknown solution."
    default = 5
  "--initial-distance"
    arg_type = Float64
    help = "The normalized initial distance from the solution set."
    default = 1.0
  "--eps-decrease"
    arg_type = Float64
    help = "The multiplicative decrease factor for the loss."
    default = 0.5
  "--eps-distance"
    arg_type = Float64
    help = "A multiplicative factor η for the distance between the initial " *
           "point `y₀` and the output `y` of the bundle Newton method. It " *
           "requires that |y - y₀| < η * f(y₀)."
    default = 1.5
  "--seed"
    arg_type = Int
    help = "The seed for the random number generator."
    default = 123
  "--show-amortized"
    help = "Set to plot the residual vs. the amortized number of oracle calls."
    action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(args["m"], args["d"], args["k"], args["initial-distance"],
               args["eps-decrease"], args["eps-distance"],
               args["show-amortized"])
