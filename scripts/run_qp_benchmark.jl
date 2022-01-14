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
  superpolyak_solve_time::Float64
  scs_iter::Float64
  superpolyak_iter::Float64
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
  oracle_calls_limit::Int,
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
      oracle_calls_limit,
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
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  ϵ_rel,
  η_est,
  η_lb,
  bundle_budget,
  oracle_calls_limit,
  no_amortized,
)
  instance_name = filename_noext(filename)
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
  # Stop and return nothing if problem is infeasible.
  if (scs_result.status == "UNSOLVED")
    @info "Instance $(instance_name) infeasible -- terminating"
    return nothing
  end
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
    joinpath(output_folder, "superpolyak_$(filename_noext(filename)).csv"),
    result,
    no_amortized,
  )
  return SolveStat(
    filename_noext(filename),
    scs_result.solve_time,
    df_bundle.cumul_elapsed_time[end],
    scs_result.iter,
    df_bundle.cumul_oracle_calls[end],
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
  default = 1000
  "--oracle-calls-limit"
  arg_type = Int
  help = "The total number of oracle calls allowed."
  default = 100000
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
  args["oracle-calls-limit"],
  args["no-amortized"],
)
