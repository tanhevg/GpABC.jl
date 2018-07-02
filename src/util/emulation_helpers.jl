function get_training_data{D<:ContinuousUnivariateDistribution}(n_design_points::Int64,
    priors::AbstractArray{D,1},
    simulator_function::Function,
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    distance_metric::Function,
    reference_data::AbstractArray{Float64,2})

    n_var_params = length(priors)

    X = zeros(n_design_points, n_var_params)
    y = zeros(n_design_points)

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)

    for j in 1:n_var_params
        X[:,j] = rand(priors[j], n_design_points)
    end

    for i in 1:n_design_points
        model_output = simulator_function(X[i,:])
        y[i] = distance_metric(summary_statistic(model_output), reference_summary_statistic)
    end

    return X, y
end

# function build_simulator_function(ode_function::Function,
#     Tspan::Tuple{Float64,Float64}, x0::AbstractArray{Float64,1};
#     solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4(), kwargs...)

#     function simulator_function(params::AbstractArray{Float64,1})
#         prob = ODEProblem(ode_function, x0, Tspan, params)
#         return solve(prob, solver; kwargs...)
#     end
#     return simulator_function
# end
