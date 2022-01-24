using ArgParse
using CSV
using DataFrames
using Glob
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")
include("scs_util.jl")

struct SolveStat
  instance_name::String
  scs_solve_time::Float64
  scs_iter::Int
  scs_loss::Float64
  superpolyak_solve_time::Float64
  superpolyak_iter::Int
  superpolyak_loss::Float64
end

function run_benchmark(
  benchmark_folder::String,
  output_folder::String,
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
  solve_stats = Vector{SolveStat}([])
  filenames = glob(joinpath(benchmark_folder, "*.QPS"))
  for filename in filenames
    @info "Solving instance $(filename_noext(filename))"
    stat = run_experiment(
      filename,
      output_folder,
      ϵ_decrease,
      ϵ_distance,
      ϵ_tol,
      ϵ_rel,
      η_est,
      η_lb,
      bundle_budget,
      budget_weight,
      bundle_max_budget,
      bundle_step_threshold,
      oracle_calls_limit,
      exit_frequency,
      no_amortized,
    )
    # Only store statistics if instance was not infeasible.
    if !isnothing(stat)
      push!(solve_stats, stat)
    end
  end
  CSV.write("scs_vs_superpolyak.csv", DataFrame(solve_stats))
end

function run_experiment(
  filename::String,
  output_folder::String,
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
  instance_name = filename_noext(filename)
  problem = read_problem_from_qps_file(filename, :fixed)
  loss = forward_backward_error(problem)
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
  # Stop and return nothing if problem is infeasible.
  if !(scs_result.status_val in [1; 2])
    @info "Instance $(instance_name) infeasible -- terminating"
    return nothing
  end
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
  df_bundle = save_superpolyak_result(
    joinpath(output_folder, "superpolyak_$(filename_noext(filename)).csv"),
    result,
    no_amortized,
  )
  return SolveStat(
    filename_noext(filename),
    scs_result.solve_time,
    scs_result.iter,
    loss(scs_result.sol[1:(n + m)]),
    df_bundle.cumul_elapsed_time[end],
    df_bundle.cumul_oracle_calls[end],
    df_bundle.fvals[end],
  )
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on ReLU regression.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--benchmark-folder"
  arg_type = String
  help = "The path to the folder containing the benchmark."
  "--output-folder"
  arg_type = String
  help = "The path to the output folder where history will be saved."
  default = "results"
  "--eps-rel"
  arg_type = Float64
  help = "The relative tolerance used when calling SCS."
  default = 1e-15
  "--bundle-budget"
  arg_type = Int
  help = "The per-call budget of the bundle method used."
  default = 100
  "--budget-weight"
  arg_type = Float64
  help = "The weight by which to update the estimate of the bundle budget."
  default = 0.5
  "--bundle-max-budget"
  arg_type = Int
  help = "The maximum per-call budget of the bundle method used."
  default = 1000
  "--bundle-step-threshold"
  arg_type = Float64
  help = "The loss threshold below which bundle steps will be attempted."
  default = 1e-4
  "--oracle-calls-limit"
  arg_type = Int
  help = "The total number of oracle calls allowed."
  default = 100000
  "--exit-frequency"
  arg_type = Int
  help = "The exit frequency of calls to SCS."
  default = 250
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_benchmark(
  args["benchmark-folder"],
  args["output-folder"],
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
