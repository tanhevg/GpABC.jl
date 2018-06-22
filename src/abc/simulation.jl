function SimulatedABCRejection(reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold::Float64,
    priors::Vector{ContinuousUnivariateDistribution},
    summary_statistic::Union{String,Vector{String},Function},
    simulator_function::Function;
    distance_function::Function=eucildean,
    kwargs...
    )

    n_params = length(priors)

    input = SimulatedABCRejectionInput(n_params, n_particles, threshold,
                                        priors, summary_statistic,
                                        distance_function, simulator_function)

    return ABCRejection(input, reference_data; kwargs...)

end

function SimulatedABCSMC(reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold_schedule::Vector{Float64},
    priors::Vector{ContinuousUnivariateDistribution},
    summary_statistic::Union{String,Vector{String},Function},
    simulator_function::Function;
    distance_function::Function=eucildean,
    kwargs...
    )

    n_params = length(priors)

    input = SimulatedABCSMCInput(n_params, n_particles, threshold_schedule,
                                    priors, summary_statistic,
                                    distance_function, simulator_function)

    return ABCSMC(input, reference_data; kwargs...)

end
