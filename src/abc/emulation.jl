"""
    EmulatedABCRejection

A convenience function that trains a Gaussian process emulator of  type [`GPmodel](@ref) then uses it in emulation-based
rejection-ABC. It creates the training data by simulating the model for the design points, trains
the emulator, creates the [`EmulatedABCRejectionInput`](@ref) object then calls [`ABCrejection](@ref).

# Fields
- `n_design_points::Int64`: The number of parameter vectors used to train the Gaussian process emulator
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold::Float64`: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `distance_metric::Function`: Any function that computes the distance between 2 1D Arrays (optional - default is to use the Euclidean distance).
- `gpkernel::AbstractGPKernel`: An object inheriting from [`AbstractGPKernel`](@ref) that is the Gaussian process kernel. (optional - default is the ARD-RBF/squared exponential kernel).
- `batch_size::Int64`: The number of predictions to be made in each batch (optional - default is 10 ``\\times`` `n_particles`).
- `max_iter::Int64`: The maximum number of iterations/batches before termination.
- `kwargs`: optional keyword arguments passed to ['ABCrejection'](@ref).
"""
function EmulatedABCRejection(n_design_points::Int64,
    reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold::Float64, priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    emulation_type::AbstractEmulatorTraining = DefaultEmulatorTraining(),
    distance_metric::Function=Distances.euclidean,
    # gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
    batch_size::Int64=10*n_particles, max_iter::Int64=1000,
    emulator_training::ET = DefaultEmulatorTraining(),
    # repetitive_training::RepetitiveTraining=RepetitiveTraining(),
    kwargs...) where {
    D<:ContinuousUnivariateDistribution,
    ET<:AbstractEmulatorTraining
    }

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)

    emulator_training_input = EmulatorTrainingInput(
        DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_metric),
        n_design_points,
        emulator_training)

    # gp_train_function = function(prior_sampling_function::Function)
    #     abc_train_emulator(prior_sampling_function,
    #             n_design_points,
    #             reference_summary_statistic,
    #             simulator_function,
    #             summary_statistic,
    #             distance_metric,
    #             emulation_type,
    #             repetitive_training,
    #             )
    # end
    #
    input = EmulatedABCRejectionInput(length(priors), n_particles, threshold,
        priors, batch_size, max_iter, emulator_training_input)

    return ABCrejection(input, reference_data; kwargs...)
end

"""
    EmulatedABCSMC

A convenience function that trains a Gaussian process emulator of type [`GPmodel](@ref) then uses it in emulation-based
ABC-SMC. It creates the training data by simulating the model for the design points, trains
the emulator, creates the [`EmulatedABCSMCInput`](@ref) object then calls [`ABCSMC`](@ref).

# Fields
- `n_design_points::Int64`: The number of parameter vectors used to train the Gaussian process emulator
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarise model output. REFER TO DOCS
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `distance_metric::Function`: Any function that computes the distance between 2 1D Arrays (optional - default is to use the Euclidean distance).
- `gpkernel::AbstractGPKernel`: An object inheriting from [`AbstractGPKernel`](@ref) that is the Gaussian process kernel. (optional - default is the ARD-RBF/squared exponential kernel).
- `batch_size::Int64`: The number of predictions to be made in each batch (optional - default is 10 ``\\times`` `n_particles`).
- `max_iter::Int64`: The maximum number of iterations/batches before termination.
- `kwargs`: optional keyword arguments passed to ['ABCSMC'](@ref).
"""
function EmulatedABCSMC(
    n_design_points::Int64,
    reference_data::AbstractArray{Float64,2},
    n_particles::Int64,
    threshold_schedule::AbstractArray{Float64,1},
    priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    emulator_training::AbstractEmulatorTraining = DefaultEmulatorTraining(),
    distance_metric::Function=Distances.euclidean,
    # gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
    batch_size::Int64=10*n_particles, max_iter::Int64=20,
    repetitive_training::RepetitiveTraining=RepetitiveTraining(),
    kwargs...) where {
    D<:ContinuousUnivariateDistribution
    }

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)

    gp_train_function = function(prior_sampling_function)
        abc_train_emulator(prior_sampling_function,
                n_design_points,
                reference_summary_statistic,
                simulator_function,
                summary_statistic,
                distance_metric,
                emulator_training,
                repetitive_training,
                )
    end

    input = EmulatedABCSMCInput(length(priors), n_particles, threshold_schedule,
        priors, batch_size, max_iter, gp_train_function)

    return ABCSMC(input, reference_data; kwargs...)
end

"""
    model_selection

Perform model selection using emulation-based ABC.

# Arguments
- `n_design_points::Int64`: The number of parameter vectors used to train the Gaussian process emulator.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `parameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution},1}`: Priors for the parameters of each model. The length of the outer array is the number of models.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_functions::AbstractArray{Function,1}`: An array of functions that take a parameter vector as an argument and outputs model results (one per model).
- 'model_prior::DiscreteUnivariateDistribution': The prior from which models are sampled. Default is a discrete, uniform distribution.
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Integer`: The maximum number of simulations that will be run. The default is 1000*`n_particles`. Each iteration samples a single model and performs ABC using a single particle.
- `max_batch_size::Integer`: The maximum batch size for the emulator when making predictions.

# Returns
A ['ModelSelectionOutput'](@ref) object that contains which models are supported by the observed data.
"""
function model_selection(
    n_design_points::Int64,
    reference_data::AbstractArray{Float64,2},
    n_particles::Int64,
    threshold_schedule::AbstractArray{Float64,1},
    parameter_priors::AbstractArray{AD,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_functions::AbstractArray{Function,1};
    model_prior::DiscreteUnivariateDistribution=Distributions.DiscreteUniform(1,length(parameter_priors)),
    distance_function::Function=Distances.euclidean,
    max_iter::Integer=1000,
    max_batch_size::Integer=1000) where {
    D<:ContinuousUnivariateDistribution,
    AD<:AbstractArray{D,1}
    }

    #
    # A set of functions that return a trained emulator with a prior sampling function as an argument
    #
    emulator_trainers = [f(prior_sampler) = abc_train_emulator(prior_sampler,
            n_design_points,
            build_summary_statistic(summary_statistic)(reference_data),
            sim,
            build_summary_statistic(summary_statistic),
            distance_function)
        for sim in simulator_functions]

    input = EmulatedModelSelectionInput(length(parameter_priors), n_particles, threshold_schedule, model_prior,
        parameter_priors, emulator_trainers, max_batch_size, max_iter)

    return model_selection(input, reference_data)
end
