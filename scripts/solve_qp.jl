using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")
include("scs_util.jl")

function run_experiment(
  filename::String,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  ϵ_rel,
  η_est,
  η_lb,
  bundle_system_solver,
  bundle_budget,
  oracle_calls_limit,
  no_amortized,
)
  problem = read_problem_from_qps_file(filename, :fixed)
  m, n = size(problem.A)
  z_init = zeros(2m + n)
  @info "Running SCS..."
  scs_result = solve_with_scs(
    problem,
    zeros(n),
    zeros(m),
    zeros(m),
    ϵ_tol = ϵ_tol,
    ϵ_rel = ϵ_rel,
    use_direct_solver = true,
    iteration_limit = oracle_calls_limit,
  )
  @info "Running SuperPolyak..."
  result = superpolyak_with_scs(
    problem,
    z_init,
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    ϵ_rel = ϵ_rel,
    η_est = η_est,
    η_lb = η_lb,
    oracle_calls_limit = oracle_calls_limit,
    bundle_budget = bundle_budget,
  )
  df_bundle = save_superpolyak_result(
    "qp_bundle.csv",
    result,
    no_amortized,
  )
  @info "SCS iters: $(scs_result.iter) - SuperPolyak iters: $(df_bundle.cumul_oracle_calls[end])"
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on ReLU regression.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--filename"
  arg_type = String
  help = "The path to the .QPS file containing the QP."
  "--bundle-budget"
  arg_type = Int
  help = "The per-call budget of the bundle method used."
  default = 1000
  "--oracle-calls-limit"
  arg_type = Int
  help = "The total number of oracle calls allowed."
  default = 100000
  "--eps-rel"
  arg_type = Float64
  help = "The relative tolerance used in calculations."
  default = 1e-15
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["filename"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eps-rel"],
  args["eta-est"],
  args["eta-lb"],
  args["bundle-system-solver"],
  args["bundle-budget"],
  args["oracle-calls-limit"],
  args["no-amortized"],
)
