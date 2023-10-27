using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function integer_list_to_str(v::Vector{Int})
  return foldl((l, r) -> l * "-" * r, string.(v))
end

function run_experiment(
  m,
  k,
  hidden_layers,
  δ,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  bundle_system_solver,
  no_amortized,
  plot_inline,
)
  problem = SuperPolyak.generative_sensing_problem(m, k, hidden_layers)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  z_init = SuperPolyak.initializer(problem, δ)
  # Run grad_fn() once to avoid compilation latency
  @info "Precompiling grad_fn..."
  grad_fn(z_init)
  @info "Running subgradient method..."
  _, loss_history_polyak, oracle_calls_polyak, elapsed_time_polyak =
    SuperPolyak.subgradient_method(loss_fn, grad_fn, z_init[:], ϵ_tol)
  df_polyak = DataFrame(
    t = 1:length(loss_history_polyak),
    fvals = loss_history_polyak,
    cumul_oracle_calls = 0:oracle_calls_polyak,
    cumul_elapsed_time = cumsum(elapsed_time_polyak),
  )
  CSV.write(
    "generative_sensing_$(m)_$(k)_$(integer_list_to_str(hidden_layers))_polyak.csv",
    df_polyak,
  )
  @info "Running SuperPolyak..."
  result = SuperPolyak.superpolyak(
    loss_fn,
    grad_fn,
    z_init,
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    bundle_system_solver = bundle_system_solver,
  )
  df_bundle = save_superpolyak_result(
    "generative_sensing_$(m)_$(k)_$(integer_list_to_str(hidden_layers))_bundle.csv",
    result,
    no_amortized,
  )
  if plot_inline
    semilogy(df_bundle.cumul_oracle_calls, df_bundle.fvals, "bo--")
    semilogy(0:oracle_calls_polyak, loss_history_polyak, "r--")
    legend(["SuperPolyak", "PolyakSGM"])
    xlabel("Oracle calls")
    ylabel(L"$ f(x_k) - f^* $")
    show()
  end
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on generative model sensing.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--k"
  arg_type = Int
  help = "The dimension of the unknown signal."
  default = 10
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 150
  "--hidden-layers"
  arg_type = Int
  nargs = '+'
  help = "The hidden layer size, from last to first."
  default = [600, 250]
  "--plot-inline"
  help = "Set to plot the results after running the script."
  action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["m"],
  args["k"],
  args["hidden-layers"],
  args["initial-distance"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eta-est"],
  args["eta-lb"],
  args["bundle-system-solver"],
  args["no-amortized"],
  args["plot-inline"],
)
