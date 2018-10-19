using GpABC, DifferentialEquations, Distances, Distributions

# srand(4)

#
# ABC settings
#

n_particles = 3000
threshold_schedule = [3.0, 2.0, 1.0, 0.5, 0.2]
# threshold_schedule = [3.0, 2.0, 1.0]
distance_metric = euclidean
summary_stats = GpABC.keep_all_summary_statistic
progress_every = 1000

#
# Emulation settings
#
n_design_points = 300

#
# True parameters
#
true_params =  [2.0, 2.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
priors = [Uniform(0.2, 5.), Uniform(0.2, 5.), Uniform(10., 20.),
            Uniform(0.2, 2.), Uniform(0.2, 2.), Uniform(0.2, 2.),
            Uniform(75., 125.),
            Uniform(0.2, 2.), Uniform(0.2, 2.), Uniform(0.2, 2.)]
param_indices = [1,2,3,4, 5, 6]
n_var_params = length(param_indices)


#
# ODE solver settings
#
Tspan = (0.0, 10.0)
x0 = [0.0, 0.0, 0.0]
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

function simulator_function(var_param_idx::Int, var_param::Real)
    params = copy(true_params)
    params[var_param_idx] = var_param
    GeneReg(params, Tspan, x0, solver, saveat)
end

function simulator_function(var_param_indices::AbstractArray{Int, 1}, var_params::AbstractArray{T, 1}) where {T<:Real}
    params = copy(true_params)
    params[var_param_indices] .= var_params
    GeneReg(params, Tspan, x0, solver, saveat)
end

function simulator_function(var_params::AbstractArray{T, 1}) where {T<:Real}
    simulator_function(param_indices, var_params)
    params = copy(true_params)
    params[param_indices] .= var_params
    GeneReg(params, Tspan, x0, solver, saveat)
end

function simulate_distance(var_params)
    distance_metric(summary_stats(reference_data),
        summary_stats(simulator_function(var_params)))
end

function simulate_distance(var_param_idx::Int, var_param::Real)
    distance_metric(summary_stats(reference_data),
        summary_stats(simulator_function(var_param_idx, var_param)))
end

function simulate_distance(var_param_indices::AbstractArray{Int, 1}, var_params::AbstractArray{T, 1}) where {T<:Real}
    distance_metric(summary_stats(reference_data),
        summary_stats(simulator_function(var_param_indices, var_params)))
end

println("SIMULATION")
sim_out = SimulatedABCSMC(reference_data, n_particles, threshold_schedule,
    priors[param_indices], summary_stats, simulator_function)

println("EMULATION")
emu_out = EmulatedABCSMC(n_design_points, reference_data, n_particles, threshold_schedule,
    priors[param_indices], summary_stats, simulator_function,
    # emulator_retraining = IncrementalRetraining(10, 100)
    emulator_retraining = DiscardPriorRetraining()
    # repetitive_training=RepetitiveTraining(rt_iterations=3, rt_extra_training_points=5),
    )

# TODO finish
#=
import GpABC.abc_retrain_emulator

struct PreviousPopulationThresholdRetraining
    n_design_points::Int
    n_below_threshold::Int
    max_iter::Int
end

function abc_retrain_emulator(
    gpm::GPModel,
    particle_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::PreviousPopulationThresholdRetraining
    ) where {T<:Real}
    n_below_threshold = 0
    n_above_threshold = 1
    idx = 1
    n_iter = 0
    training_x = zeros(retraining_settings.n_design_points, size(gpm.gp_training_x, 2))
    training_y = zeros(retraining_settings.n_design_points, 1)
    while(n_below_threshold < retraining_settings.n_below_threshold
        && idx <= retraining_settings.n_design_points
        && n_iter < retraining_settings.max_iter)
        sample_x = particle_sampling_function(n_design_points)
        sample_y = simulate_distance(training_x, training_input.distance_simulation_input)
        idx_below_threshold = find(sample_y .<= epsilon)
        if length(idx_below_threshold) > retraining_settings.n - n_below_threshold
            idx_below_threshold = idx_below_threshold[1:size(training_x, 1) - n_below_threshold]
        end
        training_x[idx:idx+legth(idx_below_threshold)-1] .= sample_x[idx_below_threshold, :]
        training_y[idx:idx+legth(idx_below_threshold)-1] .= sample_y[idx_below_threshold]
        idx += length(idx_below_threshold)
        n_below_threshold += length(idx_below_threshold)

        n_iter += 1

    end
    n_design_points = size(gpm.gp_training_x, 1)
    training_x = particle_sampling_function(n_design_points)
    training_y = simulate_distance(training_x, training_input.distance_simulation_input)
    train_emulator(training_x, reshape(training_y, n_design_points, 1), training_input.emulator_training)
end
=#
