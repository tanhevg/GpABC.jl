"""
    EmulatedABCRejection(
        reference_data,
        simulator_function,
        priors,
        threshold,
        n_particles,
        n_design_points;
        summary_statistic               = "keep_all",
        distance_function               = Distances.euclidean,
        batch_size                      = 10*n_particles,
        max_iter                        = 1000,
        emulator_training               = DefaultEmulatorTraining(),
        emulated_particle_selection     = MeanEmulatedParticleSelection(),
        write_progress                  = true,
        progress_every                  = 1000,
        )

Run emulation-based rejection ABC algorithm. Model simulation results are used to train the
emulator, which is then used to get the approximated distance for each particle. The rest
of the workflow is similar to [`SimulatedABCRejection`](@ref).

See [ABC Overview](@ref abc-overview) for more details.

# Mandatory arguments
- `reference_data::AbstractArray{Float,2}`: Observed data to which the simulated model output will be compared. Array dimensions sould match that of the simulator function result.
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: Continuous univariate distributions, from which candidate parameters will be sampled. Array size should match the number of parameters.
- `threshold::Float`: The ``\\varepsilon`` threshold to be used in ABC algorithm. Only those particles that produce emulated results that are within this threshold from the reference data are included into the posterior.
- `n_particles::Int`: The number of parameter vectors (particles) that will be included in the posterior.
- `n_design_points::Int`: The number of design particles that will be simulated to traing the emulator.

# Optional keyword arguments
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Summary statistics that will be applied to the data before computing the distances. Defaults to `keep_all`. See [detailed documentation of summary statistics](@ref summary_stats).
- `distance_function::Union{Function,Metric}`: A function or metric that will be used to compute the distance between the summary statistic of the simulated data and that of reference data. Defaults to `Distances.euclidean`.
- `batch_size::Int`: The number of particles that will be emulated on each iteration. Defaults to `1000 * n_particles`.
- `max_iter::Int`: The maximum number of emulations that will be run. Defaults to 1000.
- `emulator_training<:AbstractEmulatorTraining`: This determines how the emulator will be trained. See [`AbstractEmulatorTraining`](@ref) for more details.
- `emulated_particle_selection<:AbstractEmulatedParticleSelection`: This determines how the particles that will be added to the posterior are selected after each emulation run. See [`AbstractEmulatedParticleSelection`](@ref) for details. Defaults to [`MeanEmulatedParticleSelection`](@ref).
- `write_progress::Bool`: Whether algorithm progress should be printed on standard output. Defaults to `true`.

# Returns
An [`ABCRejectionOutput`](@ref) object.
"""
function EmulatedABCRejection(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold::AF,
    n_particles::Int,
    n_design_points::Int;
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    distance_function::Union{Function,Metric}=Distances.euclidean,
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
        DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_function),
        n_design_points,
        emulator_training)

    input = EmulatedABCRejectionInput(length(priors), n_particles, threshold,
        priors, batch_size, max_iter, emulator_training_input, emulated_particle_selection)

    return ABCrejection(input; kwargs...)
end

"""
    EmulatedABCSMC(
        reference_data,
        simulator_function,
        priors,
        threshold_schedule,
        n_particles,
        n_design_points;
        summary_statistic               = "keep_all",
        distance_function               = Distances.euclidean,
        batch_size                      = 10*n_particles,
        max_iter                        = 1000,
        emulator_training               = DefaultEmulatorTraining(),
        emulator_retraining             = NoopRetraining(),
        emulated_particle_selection     = MeanEmulatedParticleSelection(),
        write_progress                  = true,
        progress_every                  = 1000,
        )

Run emulation-based ABC-SMC algorithm. This is similar to [`EmulatedABCRejection`](@ref),
the main difference being that an array of thresholds is provided instead of a single threshold.
It is assumed that thresholds are sorted in decreasing order.

An emulation based ABC iteration is executed for each threshold. For the first threshold, the provided prior is used.
For each subsequent threshold, the posterior from the previous iteration is used as a prior.

See [ABC Overview](@ref abc-overview) for more details.

# Mandatory arguments
- `reference_data::AbstractArray{Float,2}`: Observed data to which the simulated model output will be compared. Array dimensions sould match that of the simulator function result.
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: Continuous univariate distributions, from which candidate parameters will be sampled during the first iteration. Array size should match the number of parameters.
- `threshold_schedule::AbstractArray{Float,1}`: The threshold schedule to be used in ABC algorithm. An ABC iteration is executed for each threshold. It is assumed that thresholds are sorted in decreasing order.
- `n_particles::Int`: The number of parameter vectors (particles) that will be included in the posterior.
- `n_design_points::Int`: The number of design particles that will be simulated to traing the emulator.

# Optional keyword arguments
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Summary statistics that will be applied to the data before computing the distances. Defaults to `keep_all`. See [detailed documentation of summary statistics](@ref summary_stats).
- `distance_function::Union{Function,Metric}`: A function or metric that will be used to compute the distance between the summary statistic of the simulated data and that of reference data. Defaults to `Distances.euclidean`.
- `batch_size::Int`: The number of particles that will be emulated on each iteration. Defaults to `1000 * n_particles`.
- `max_iter::Int`: The maximum number of emulations that will be run. Defaults to 1000.
- `emulator_training<:AbstractEmulatorTraining`: This determines how the emulator will be trained for each iteration. See [`AbstractEmulatorTraining`](@ref) for more details.
- `emulator_retraining<:AbstractEmulatorRetraining`: This is used to specify parameters of additional emulator retraining that can be done for each iteration. By default this retraining is switched off ([`NoopRetraining`](@ref)). See [`AbstractEmulatorRetraining`] for more details.
- `emulated_particle_selection<:AbstractEmulatedParticleSelection`: This determines how the particles that will be added to the posterior are selected after each emulation run. See [`AbstractEmulatedParticleSelection`](@ref) for details. Defaults to [`MeanEmulatedParticleSelection`](@ref).
- `write_progress::Bool`: Whether algorithm progress should be printed on standard output. Defaults to `true`.

# Returns
An [`ABCSMCOutput`](@ref) object.
"""
function EmulatedABCSMC(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int,
    n_design_points::Int;
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    emulator_training::ET = DefaultEmulatorTraining(),
    distance_metric::Union{Function,Metric}=Distances.euclidean,
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

    return ABCSMC(input; kwargs...)
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
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. Defaults to `keep_all`. See [detailed documentation of summary statistics](@ref summary_stats).
- `simulator_functions::AbstractArray{Function,1}`: An array of functions that take a parameter vector as an argument and outputs model results (one per model).
- 'model_prior::DiscreteUnivariateDistribution': The prior from which models are sampled. Default is a discrete, uniform distribution.
- `distance_function::Union{Function,Metric}`: A function or metric that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Int`: The maximum number of simulations that will be run. The default is 1000*`n_particles`. Each iteration samples a single model and performs ABC using a single particle.
- `max_batch_size::Int`: The maximum batch size for the emulator when making predictions.

# Returns
A [`ModelSelectionOutput`](@ref) object that contains which models are supported by the observed data.
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
    distance_function::Union{Function,Metric}=Distances.euclidean,
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
