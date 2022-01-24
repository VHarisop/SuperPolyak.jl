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
  bundle_step_threshold::Float64,
  oracle_calls_limit::Int,
  exit_frequency::Int,
  no_amortized::Bool,
)
  problem = read_problem_from_qps_file(filename, :fixed)
  m, n = size(problem.A)
  @info "Running SCS..."
  scs_result = superpolyak_with_scs(
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
    bundle_budget = 0,
    budget_weight = 0.0,
    bundle_max_budget = 0,
    bundle_step_threshold = ϵ_tol,
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
    bundle_step_threshold = bundle_step_threshold,
  )
  df_scs = save_superpolyak_result(
    "qp_$(filename_noext(filename))_scs.csv",
    scs_result,
    no_amortized,
  )
  df_bundle = save_superpolyak_result(
    "qp_$(filename_noext(filename))_bundle.csv",
    result,
    no_amortized,
  )
  @info "SCS iters: $(df_scs.cumul_oracle_calls[end]) " *
        "- SuperPolyak iters: $(df_bundle.cumul_oracle_calls[end])"
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
  "--bundle-step-threshold"
  arg_type = Float64
  help = "The loss threshold below which bundle steps will be triggered."
  default = 1e-4
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
  args["bundle-step-threshold"],
  args["oracle-calls-limit"],
  args["exit-frequency"],
  args["no-amortized"],
)
