function SimulatedABCRejection{D<:ContinuousUnivariateDistribution}(reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold::Float64,
    priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_function::Function=Distances.euclidean,
    kwargs...
    )

    n_params = length(priors)

    input = SimulatedABCRejectionInput(n_params, n_particles, threshold,
                                        priors, summary_statistic,
                                        distance_function, simulator_function)

    return ABCrejection(input, reference_data; kwargs...)

end

function SimulatedABCSMC{D<:ContinuousUnivariateDistribution}(reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold_schedule::AbstractArray{Float64,1},
    priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_function::Function=Distances.euclidean,
    kwargs...
    )

    n_params = length(priors)

    input = SimulatedABCSMCInput(n_params, n_particles, threshold_schedule,
                                    priors, summary_statistic,
                                    distance_function, simulator_function)

    return ABCSMC(input, reference_data; kwargs...)

end
