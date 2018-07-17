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
function EmulatedABCRejection{D<:ContinuousUnivariateDistribution}(n_design_points::Int64,
    reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold::Float64, priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_metric::Function=Distances.euclidean,
    gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
    batch_size::Int64=10*n_particles, max_iter::Int64=1000,
    repetitive_training_settings::RepetitiveTrainingSettings=RepetitiveTrainingSettings(),
    kwargs...)

    n_var_params = length(priors)

    X = sample_from_priors(n_design_points, priors)
    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    y = simulate_distance(X, simulator_function, summary_statistic, distance_metric, reference_summary_statistic)

    gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    gp_train(gpem)
    emulator_retraining_function = function(gpem)
        retrain_emulator(repetitive_training_settings, gpem, priors,
            simulator_function, summary_statistic, distance_metric,
            reference_summary_statistic)
    end
    emulate_distance_function(params, em) = gp_regression_sample(params, em)

    input = EmulatedABCRejectionInput(n_var_params, n_particles, threshold,
        priors, emulator_retraining_function, emulate_distance_function, batch_size, max_iter, gpem)

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
function EmulatedABCSMC{D<:ContinuousUnivariateDistribution}(n_design_points::Int64,
    reference_data::AbstractArray{Float64,2}, n_particles::Int64,
    threshold_schedule::AbstractArray{Float64,1}, priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_metric::Function=Distances.euclidean,
    gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
    batch_size::Int64=10*n_particles, max_iter::Int64=1000,
    repetitive_training_settings::RepetitiveTrainingSettings=RepetitiveTrainingSettings(),
    kwargs...)

    n_var_params = length(priors)

    X = sample_from_priors(n_design_points, priors)
    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    y = simulate_distance(X, simulator_function, summary_statistic, distance_metric, reference_summary_statistic)

    gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    gp_train(gpem)
    emulator_retraining_function = function(gpem)
        retrain_emulator(repetitive_training_settings, gpem, priors,
            simulator_function, summary_statistic, distance_metric,
            reference_summary_statistic)
    end
    emulate_distance_function(params, em) = gp_regression_sample(params, em)

    input = EmulatedABCSMCInput(n_var_params, n_particles, threshold_schedule,
        priors, emulator_retraining_function, emulate_distance_function, batch_size, max_iter, gpem)

    return ABCSMC(input, reference_data; kwargs...)
end
