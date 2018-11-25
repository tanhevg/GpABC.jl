"""
    SimulatedABCRejection

Run a simulation-based ABC-rejection computation. This is a convenience wrapper that constructs a
[`SimulatedABCRejectionInput`](@ref) object then calls ['ABCrejection'](@ref).

# Arguments
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Int64`: The maximum number of simulations that will be run. The default is 1000*`n_particles`.
- `kwargs`: optional keyword arguments passed to ['ABCrejection'](@ref).

# Returns
A ['SimulatedABCRejectionOutput'](@ref) object.
"""
function SimulatedABCRejection(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold::AF,
    n_particles::Int;
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    distance_function::Function=Distances.euclidean,
    max_iter::Int=10 * n_particles,
    kwargs...
    ) where {
    AF<:AbstractFloat,
    D<:ContinuousUnivariateDistribution
    }

    n_params = length(priors)
    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    distance_simulation_input = DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_function)
    input = SimulatedABCRejectionInput(n_params, n_particles, threshold,
                                        priors,
                                        distance_simulation_input,
                                        max_iter)

    return ABCrejection(input; kwargs...)

end

"""
    SimulatedABCSMC

Run a emulation-based ABC-rejection computation. This is a convenience wrapper that constructs a
[`SimulatedABCSMCInput`](@ref) object then calls ['ABCrejection'](@ref).

# Arguments
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Int`: The maximum number of simulations that will be run. The default is 1000*`n_particles`.
- `kwargs`: optional keyword arguments passed to ['ABCrejection'](@ref).

# Returns
A ['SimulatedABCSMCOutput'](@ref) object that contains the posteriors at each ABC-SMC population and other information.
"""
function SimulatedABCSMC(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int;
    summary_statistic::Union{String,AbstractArray{String,1},Function} = "keep_all",
    distance_function::Function=Distances.euclidean,
    max_iter::Int=10 * n_particles,
    kwargs...
    ) where {
    AF<:AbstractFloat,
    D<:ContinuousUnivariateDistribution
    }

    n_params = length(priors)

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    distance_simulation_input = DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_function)
    input = SimulatedABCSMCInput(n_params, n_particles, threshold_schedule,
                                    priors, distance_simulation_input,
                                    max_iter)

    return ABCSMC(input; kwargs...)

end

"""
    SimulatedModelSelection

Perform model selection using simulation-based ABC.

# Arguments
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `parameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution},1}`: Priors for the parameters of each model. The length of the outer array is the number of models.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_functions::AbstractArray{Function,1}`: An array of functions that take a parameter vector as an argument and outputs model results (one per model).
- 'model_prior::DiscreteUnivariateDistribution': The prior from which models are sampled. Default is a discrete, uniform distribution.
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Int`: The maximum number of simulations that will be run. The default is 1000*`n_particles`. Each iteration samples a single model and performs ABC using a single particle.

# Returns
A ['ModelSelectionOutput'](@ref) object that contains which models are supported by the observed data.
"""
function SimulatedModelSelection(
    reference_data::AbstractArray{AF,2},
    simulator_functions::AbstractArray{Function,1},
    parameter_priors::AbstractArray{AD,1},
    threshold_schedule::AbstractArray{Float64,1},
    n_particles::Int,
    model_prior::DiscreteUnivariateDistribution=Distributions.DiscreteUniform(1,length(parameter_priors));
    summary_statistic::Union{String,AbstractArray{String,1},Function}="keep_all",
    distance_function::Function=Distances.euclidean,
    max_iter::Int=10000
    ) where {
    AF<:AbstractFloat,
    D<:ContinuousUnivariateDistribution,
    AD<:AbstractArray{D,1}
    }

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    distance_simulation_input = [DistanceSimulationInput(reference_summary_statistic,
        simulator_functions[m], summary_statistic, distance_function) for m in 1:length(parameter_priors)]
    input = SimulatedModelSelectionInput(length(parameter_priors),
        n_particles,
        threshold_schedule,
        model_prior,
        parameter_priors,
        distance_simulation_input,
        max_iter)

    return model_selection(input)
end
