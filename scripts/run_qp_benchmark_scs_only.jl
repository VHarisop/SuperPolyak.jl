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
  solve_time::Float64
  iter::Int
  status::String
  status_val::Int
end

function run_benchmark(
  benchmark_folder::String,
  ϵ_tol::Float64,
  ϵ_rel::Float64,
  oracle_calls_limit::Int,
)
  solve_stats = Vector{SolveStat}([])
  filenames = glob(joinpath(benchmark_folder, "*.QPS"))
  for filename in filenames
    @info "Solving instance $(filename_noext(filename))"
    stat = run_experiment(
      filename,
      ϵ_tol,
      ϵ_rel,
      oracle_calls_limit,
    )
    push!(solve_stats, stat)
  end
  CSV.write("maros_meszaros_scs.csv", DataFrame(solve_stats))
end

function run_experiment(
  filename::String,
  ϵ_tol,
  ϵ_rel,
  oracle_calls_limit,
)
  instance_name = filename_noext(filename)
  problem = read_problem_from_qps_file(filename, :fixed)
  m, n = size(problem.A)
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
  status_string = ""
  if scs_result.status_val == 1
    status_string *= "SOLVED"
  elseif scs_result.status_val == 2
    status_string *= "ITERATION_LIMIT_REACHED"
  elseif scs_result.status_val in [-1; -2; -6; -7]
    status_string *= "INFEASIBLE_OR_UNBOUNDED"
  else
    status_string *= "ERROR"
  end
  return SolveStat(
    filename_noext(filename),
    scs_result.solve_time,
    scs_result.iter,
    status_string,
    scs_result.status_val,
  )
end

settings = ArgParseSettings(
  description = "Run SCS on all instances from the Maros-Meszaros benchmark."
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--benchmark-folder"
  arg_type = String
  help = "The path to the folder containing the benchmark."
  "--eps-rel"
  arg_type = Float64
  help = "The relative tolerance used when calling SCS."
  default = 1e-15
  "--oracle-calls-limit"
  arg_type = Int
  help = "The total number of oracle calls allowed."
  default = 100000
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_benchmark(
  args["benchmark-folder"],
  args["eps-tol"],
  args["eps-rel"],
  args["oracle-calls-limit"],
)
