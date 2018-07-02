function EmulatedABCRejection{D<:ContinuousUnivariateDistribution}(n_design_points::Int64,
    reference_data::AbstractArray{Float64,2},
    n_particles::Int64, threshold::Float64, priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_metric::Function=Distances.euclidean,
    gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
    batch_size::Int64=10*n_particles, max_iter::Int64=1000,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4(),
    kwargs...)

    n_var_params = length(priors)

    X, y = get_training_data(n_design_points, priors, simulator_function,
                                summary_statistic, distance_metric, reference_data)

    gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    gp_train(gpem)
    distance_prediction_function(params) = gp_regression(params, gpem)[1]

    input = EmulatedABCRejectionInput(n_var_params, n_particles, threshold,
        priors, distance_prediction_function, batch_size, max_iter)

    return ABCrejection(input, reference_data; kwargs...)
end

function EmulatedABCSMC{D<:ContinuousUnivariateDistribution}(n_design_points::Int64,
    reference_data::AbstractArray{Float64,2}, n_particles::Int64,
    threshold_schedule::AbstractArray{Float64,1}, priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    simulator_function::Function;
    distance_metric::Function=Distances.euclidean,
    gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
    batch_size::Int64=10*n_particles, max_iter::Int64=1000,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4(),
    kwargs...)

    n_var_params = length(priors)

    X, y = get_training_data(n_design_points, priors, simulator_function,
                                summary_statistic, distance_metric, reference_data)

    gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    gp_train(gpem)
    distance_prediction_function(params) = gp_regression(params, gpem)[1]

    input = EmulatedABCSMCInput(n_var_params, n_particles, threshold_schedule,
        priors, distance_prediction_function, batch_size, max_iter)

    return ABCSMC(input, reference_data; kwargs...)
end