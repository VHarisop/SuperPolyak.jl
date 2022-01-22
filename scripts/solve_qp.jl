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
  ϵ_decrease::Float64,
  ϵ_distance::Float64,
  ϵ_tol::Float64,
  ϵ_rel::Float64,
  η_est::Float64,
  η_lb::Float64,
  bundle_budget::Int,
  budget_weight::Float64,
  bundle_max_budget::Int,
  oracle_calls_limit::Int,
  exit_frequency::Int,
  no_amortized::Bool,
)
  problem = read_problem_from_qps_file(filename, :fixed)
  m, n = size(problem.A)
  @info "Running SCS..."
  scs_result = fallback_algorithm(
    problem,
    zeros(n),
    zeros(m),
    exit_frequency,
    oracle_calls_limit,
    ϵ_tol,
    ϵ_rel,
    0.1,
  )
  @info "Running SuperPolyak..."
  result = superpolyak_with_scs(
    problem,
    zeros(m + n),
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    ϵ_rel = ϵ_rel,
    η_est = η_est,
    η_lb = η_lb,
    exit_frequency = exit_frequency,
    oracle_calls_limit = oracle_calls_limit,
    bundle_budget = bundle_budget,
    budget_weight = budget_weight,
    bundle_max_budget = bundle_max_budget,
  )
  df_bundle = save_superpolyak_result(
    "qp_$(filename_noext(filename)).csv",
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
  default = 500
  "--bundle-max-budget"
  arg_type = Int
  help = "The maximal budget for the bundle method."
  default = 5000
  "--budget-weight"
  arg_type = Float64
  help = "The weight by which to update running average of bundle budget."
  default = 0.5
  "--exit-frequency"
  arg_type = Int
  help = "The frequency of exits to check the termination condition."
  default = 250
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
  args["bundle-budget"],
  args["budget-weight"],
  args["bundle-max-budget"],
  args["oracle-calls-limit"],
  args["exit-frequency"],
  args["no-amortized"],
)
