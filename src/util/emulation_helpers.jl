function get_training_data(n_design_points::Int64,
    priors::Vector{ContinuousUnivariateDistribution},
    simulator_function::Function, distance_metric::Function,
    reference_data::AbstractArray{Float64,2})

    n_var_params = length(priors)

    X = zeros(n_design_points, n_var_params)
    y = zeros(n_design_points)

    for j in 1:n_var_params
        X[:,j] = rand(priors[j], n_design_points)
    end

    for i in 1:n_design_points
        y[i] = distance_metric(simulator_function(X[i,:]), reference_data)
    end

    return X, y
end

function build_simulator_function(ode_function::Function,
    Tspan::Tuple{Float64,Float64}, x0::AbstractArray{Float64,1};
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4(), kwargs...)

    function simulator_function(params::AbstractArray{Float64,1})
        prob = ODEProblem(ode_function, x0, Tspan, params)
        return solve(prob, solver; kwargs...)
    end
    return simulator_function
end
