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
function EmulatedABCRejection(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold::AF,
    n_particles::Int,
    n_design_points::Int;
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    distance_metric::Function=Distances.euclidean,
    emulation_type::AbstractEmulatorTraining = DefaultEmulatorTraining(),
    batch_size::Int=10*n_particles,
    max_iter::Int=1000,
    emulator_training::ET = DefaultEmulatorTraining(),
    emulated_particle_selection::EPS = MeanEmulatedParticleSelection(),
    kwargs...) where {
    AF<:AbstractFloat,
    D<:ContinuousUnivariateDistribution,
    EPS<:AbstractEmulatedParticleSelection,
    ET<:AbstractEmulatorTraining
    }

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)

    emulator_training_input = EmulatorTrainingInput(
        DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_metric),
        n_design_points,
        emulator_training)

    input = EmulatedABCRejectionInput(length(priors), n_particles, threshold,
        priors, batch_size, max_iter, emulator_training_input, emulated_particle_selection)

    return ABCrejection(input; kwargs...)
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
function EmulatedABCSMC(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int,
    n_design_points::Int;
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    emulator_training::ET = DefaultEmulatorTraining(),
    distance_metric::Function=Distances.euclidean,
    batch_size::Int=10*n_particles,
    max_iter::Int=20,
    emulator_retraining::ER = NoopRetraining(),
    emulated_particle_selection::EPS = MeanEmulatedParticleSelection(),
    kwargs...) where {
    AF<:AbstractFloat,
    ET<:AbstractEmulatorTraining,
    ER<:AbstractEmulatorRetraining,
    EPS<:AbstractEmulatedParticleSelection,
    D<:ContinuousUnivariateDistribution
    }

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)

    emulator_training_input = EmulatorTrainingInput(
        DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_metric),
        n_design_points,
        emulator_training)

    input = EmulatedABCSMCInput(length(priors), n_particles, threshold_schedule,
        priors, batch_size, max_iter, emulator_training_input, emulator_retraining, emulated_particle_selection)

    return ABCSMC(input, reference_data, batch_size; kwargs...)
end

"""
    EmulatedModelSelection

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
- `max_iter::Int`: The maximum number of simulations that will be run. The default is 1000*`n_particles`. Each iteration samples a single model and performs ABC using a single particle.
- `max_batch_size::Int`: The maximum batch size for the emulator when making predictions.

# Returns
A ['ModelSelectionOutput'](@ref) object that contains which models are supported by the observed data.
"""
function EmulatedModelSelection(
    reference_data::AbstractArray{AF,2},
    simulator_functions::AbstractArray{Function,1},
    parameter_priors::AbstractArray{AD,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int,
    n_design_points::Int,
    model_prior::DiscreteUnivariateDistribution=Distributions.DiscreteUniform(1,length(parameter_priors));
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    distance_function::Function=Distances.euclidean,
    max_iter::Int=1000,
    max_batch_size::Int=1000,
    emulator_training::ET = DefaultEmulatorTraining(),
    emulator_retraining::ER = NoopRetraining(),
    emulated_particle_selection::EPS = MeanEmulatedParticleSelection(),
    ) where {
    D<:ContinuousUnivariateDistribution,
    AD<:AbstractArray{D,1},
    AF<:AbstractFloat,
    ET<:AbstractEmulatorTraining,
    ER<:AbstractEmulatorRetraining,
    EPS<:AbstractEmulatedParticleSelection,
    }

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    emulator_training_input = [EmulatorTrainingInput(
        DistanceSimulationInput(reference_summary_statistic, simulator_functions[m], summary_statistic, distance_function),
        n_design_points, emulator_training) for m in 1:length(parameter_priors)]

    input = EmulatedModelSelectionInput(length(parameter_priors),
        n_particles,
        threshold_schedule,
        model_prior,
        parameter_priors,
        emulator_training_input,
        emulator_retraining,
        emulated_particle_selection,
        max_batch_size,
        max_iter)

    return model_selection(input)
end
