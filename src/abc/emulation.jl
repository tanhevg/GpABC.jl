function EmulatedABCRejection(simulator_function::Function,
    reference_data::AbstractArray{Float64,2},
    n_design_points::Int64, n_particles::Int64,
    threshold::Float64, priors::Vector{ContinuousUnivariateDistribution},
    distance_metric::Function=euclidean,
    gpkernel::AbstractGPKernel=SquaredExponentialArdKernel,
    batch_size::Int64=10*n_particles, max_iter::Int64=100,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4(),
    write_progress::Bool=true)

    n_var_params = length(priors)

    X, y = get_training_data(n_design_points, priors, simulator_function,
                                distance_metric, reference_data)

    gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    gp_train(gpem)
    distance_prediction_function(params) = gp_regression(params, gpem)[1]

    input = EmulatedABCRejectionInput(n_var_params, n_particles, threshold,
        priors, distance_prediction_function, batch_size, max_iter)

    return ABCRejection(input, reference_data, write_progress=write_progress)
end
