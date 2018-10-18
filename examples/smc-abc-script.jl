using GpABC, DifferentialEquations, Distances, Distributions

# srand(4)

#
# ABC settings
#

n_particles = 1000
threshold_schedule = [3.0, 2.0, 1.0, 0.5, 0.2]
# threshold_schedule = [3.0, 2.0, 1.0]
distance_metric = euclidean
summary_stats = GpABC.keep_all_summary_statistic
progress_every = 1000

#
# Emulation settings
#
n_design_points = 100
batch_size = 1000
max_iter = 1000

#
# True parameters
#
true_params =  [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
priors = [Uniform(0., 5.), Uniform(0., 5.), Uniform(10., 20.),
            Uniform(0., 2.), Uniform(0., 2.), Uniform(0., 2.),
            Uniform(75., 125.),
            Uniform(0., 2.), Uniform(0., 2.), Uniform(0., 2.)]
# param_indices = [4,5,6]
# param_indices = [2, 3, 9, 10]
# param_indices = [1, 2, 3]
param_indices = [1, 2, 3, 7]
n_var_params = length(param_indices)


#
# ODE solver settings
#
Tspan = (0.0, 10.0)
x0 = [3.0, 2.0, 1.0]
solver = RK4()
saveat = 0.1

function ODE_3GeneReg(dx, x, pars, t)
  @. dx[1] = pars[1]/(1+pars[7]*x[3]) - pars[4]*x[1]
  @. dx[2] = pars[2]*pars[8]*x[1]/(1+pars[8]*x[1]) - pars[5]*x[2]
  @. dx[3] = pars[3]*pars[9]*x[1]*pars[10]*x[2]/((1+pars[9]*x[1])*(1+pars[10]*x[2])) - pars[6]*x[3]
end

#
# Returns the solution to the toy model as solved by DifferentialEquations
#
function GeneReg(params::AbstractArray{Float64,1},
    Tspan::Tuple{Float64,Float64}, x0::AbstractArray{Float64,1},
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64)

  prob = ODEProblem(ODE_3GeneReg, x0 ,Tspan, params)
  Obs = solve(prob, solver, saveat=saveat)

  return Obs
end

reference_data = GeneReg(true_params, Tspan, x0, solver, saveat)
# reference_data += randn(size(reference_data)) * 0.1
# simulator_function(var_params) = GeneReg(vcat(var_params, true_params[n_var_params+1:end]), Tspan, x0, solver, saveat)
function simulator_function(var_params)
    params = copy(true_params)
    params[param_indices] .= var_params
    GeneReg(params, Tspan, x0, solver, saveat)
    # ret = GeneReg(params, Tspan, x0, solver, saveat)
    # noise = randn(size(ret)) * 0.05
    # return ret + noise
end

function simulate_distance(var_params)
    distance_metric(summary_stats(reference_data),
        summary_stats(simulator_function(var_params)))
end

println("SIMULATION")
sim_out = SimulatedABCSMC(reference_data, n_particles, threshold_schedule,
    priors[param_indices], summary_stats, simulator_function)

println("EMULATION")
emu_out = EmulatedABCSMC(n_design_points, reference_data, n_particles, threshold_schedule,
    priors[param_indices], summary_stats, simulator_function,
    # emulator_retraining_settings = NoopRetrainingSettings()
    # repetitive_training=RepetitiveTraining(rt_iterations=3, rt_extra_training_points=5),
    )
