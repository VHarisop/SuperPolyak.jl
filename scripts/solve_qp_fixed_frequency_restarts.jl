using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")
include("scs_util.jl")

function solve_with_restart(
  problem::QuadraticProgram,
  x_prev::Vector{Float64},
  y_prev::Vector{Float64},
  z_prev::Vector{Float64},
  exit_frequency::Int,
  oracle_calls_limit::Int,
  ϵ_tol::Float64,
  ϵ_rel::Float64,
)
  m, n = size(problem.A)
  iter_total = 0
  scale = 0.1
  while iter_total < oracle_calls_limit
    scs_result = solve_with_scs(
      problem,
      copy(x_prev),
      copy(y_prev),
      copy(z_prev),
      ϵ_tol = ϵ_tol,
      ϵ_rel = ϵ_rel,
      use_direct_solver = true,
      iteration_limit = exit_frequency,
      initial_scale = scale,
    )
    iter_total += scs_result.iter
    scale = scs_result.scale
    if (scs_result.status_val == 2)
      copyto!(x_prev, scs_result.sol[1:n])
      copyto!(y_prev, scs_result.sol[(n+1):(n+m)])
      copyto!(z_prev, scs_result.sol[(n+m+1):end])
    else
      return iter_total
    end
  end
  return iter_total
end

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
  iters_with_restarts = solve_with_restart(
    problem,
    zeros(n),
    zeros(m),
    zeros(m),
    exit_frequency,
    oracle_calls_limit,
    ϵ_tol,
    ϵ_rel,
  )
  @info "SCS iters: $(scs_result.iter) - SuperPolyak iters: $(iters_with_restarts)"
end

settings = ArgParseSettings(
  description = "Compare full-fledged SCS against a version which exits periodically."
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
