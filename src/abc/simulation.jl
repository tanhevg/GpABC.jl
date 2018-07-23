"""
    SimulatedABCRejection

Run a simulation-based ABC-rejection computation. This is a convenience wrapper that constructs a
[`SimulatedABCRejectionInput`](@ref) object then calls ['ABCrejection'](@ref).

# Fields
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Int64`: The maximum number of simulations that will be run. The default is 1000*`n_particles`.
- `kwargs`: optional keyword arguments passed to ['ABCrejection'](@ref).
"""
function SimulatedABCRejection{D<:ContinuousUnivariateDistribution}(reference_data::AbstractArray{Float64,2},
    n_particles::Int64,
    threshold::Float64,
    priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_function::Function=Distances.euclidean,
    max_iter::Int64=1000*n_particles,
    kwargs...
    )

    n_params = length(priors)

    input = SimulatedABCRejectionInput(n_params, n_particles, threshold,
                                        priors, summary_statistic,
                                        distance_function, simulator_function,
                                        max_iter)

    return ABCrejection(input, reference_data; kwargs...)

end

"""
    SimulatedABCSMC

Run a emulation-based ABC-rejection computation. This is a convenience wrapper that constructs a
[`SimulatedABCSMCInput`](@ref) object then calls ['ABCrejection'](@ref).

# Fields
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).
- `max_iter::Int64`: The maximum number of simulations that will be run. The default is 1000*`n_particles`.
- `kwargs`: optional keyword arguments passed to ['ABCrejection'](@ref).
"""
function SimulatedABCSMC{D<:ContinuousUnivariateDistribution}(reference_data::AbstractArray{Float64,2},
    n_particles::Int64,
    threshold_schedule::AbstractArray{Float64,1},
    priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_function::Function=Distances.euclidean,
    max_iter::Int64=1000*n_particles,
    kwargs...
    )

    n_params = length(priors)

    input = SimulatedABCSMCInput(n_params, n_particles, threshold_schedule,
                                    priors, summary_statistic,
                                    distance_function, simulator_function,
                                    max_iter)

    return ABCSMC(input, reference_data; kwargs...)

end
