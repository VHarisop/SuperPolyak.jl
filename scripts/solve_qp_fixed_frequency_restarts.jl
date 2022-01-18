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
  ϵ_tol::Float64,
  ϵ_rel::Float64,
  oracle_calls_limit::Int,
  exit_frequency::Int,
)
  problem = read_problem_from_qps_file(filename, :fixed)
  m, n = size(problem.A)
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
  @info "Running SCS with fixed frequency exits"
  scs_result_with_restarts = fallback_algorithm(
    problem,
    zeros(n),
    zeros(m),
    exit_frequency,
    oracle_calls_limit,
    ϵ_tol,
    ϵ_rel,
    0.1,
  )
  @info "SCS iters: no_restarts = $(scs_result.iter) - restarts = $(scs_result_with_restarts.iter)"
end

settings = ArgParseSettings(
  description = "Compare full-fledged SCS against a version which exits " *
                "periodically to check if the forward-backward residual " *
                "has fallen below eps_tol."
)
@add_arg_table! settings begin
  "--filename"
  arg_type = String
  help = "The path to the .QPS file containing the QP."
  "--exit-frequency"
  arg_type = Int
  help = "The exit frequency for the SCS solver."
  default = 50
  "--oracle-calls-limit"
  arg_type = Int
  help = "The total number of oracle calls allowed."
  default = 100000
  "--eps-tol"
  arg_type = Float64
  help = "The absolute tolerance used in calculations."
  default = 1e-6
  "--eps-rel"
  arg_type = Float64
  help = "The relative tolerance used in calculations."
  default = 1e-15
end

args = parse_args(settings)
run_experiment(
  args["filename"],
  args["eps-tol"],
  args["eps-rel"],
  args["oracle-calls-limit"],
  args["exit-frequency"],
)
