function sample_from_priors{D<:ContinuousUnivariateDistribution}(n_design_points::Int64, priors::AbstractArray{D,1})
    n_var_params = length(priors)
    X = zeros(n_design_points, n_var_params)
    for j in 1:n_var_params
        X[:,j] = rand(priors[j], n_design_points)
    end
    X
end

function simulate_distance(parameters::AbstractArray{Float64, 2},
        simulator_function::Function,
        summary_statistic::Function,
        distance_metric::Function,
        reference_summary_statistic)
    n_design_points = size(parameters, 1)
    y = zeros(n_design_points)
    for i in 1:n_design_points
        model_output = simulator_function(parameters[i,:])
        y[i] = distance_metric(summary_statistic(model_output), reference_summary_statistic)
    end
    y
end

function retrain_emulator{D<:ContinuousUnivariateDistribution}(
        rts::RepetitiveTrainingSettings,
        gpem::GPModel,
        priors::AbstractArray{D,1},
        simulator_function::Function,
        summary_statistic::Function,
        distance_metric::Function,
        reference_summary_statistic)
    rt_count = 0
    while rt_count < rts.rt_iterations
        retraining_sample = sample_from_priors(rts.rt_sample_size, priors)
        mean, variance = gp_regression(retraining_sample, gpem)
        var_indices = sortperm(variance, rev=true)
        extra_x = retraining_sample[var_indices[1:rts.rt_extra_points], :]
        extra_y = simulate_distance(extra_x, simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
        new_training_x = vcat(gpem.gp_training_x, extra_x)
        new_training_y = vcat(gpem.gp_training_y, extra_y)
        gpem = GPModel(training_x=new_training_x, training_y=new_training_y, kernel=gpem.kernel)
        gp_train(gpem)
        rt_count += 1
    end
    gpem
end

"""
    get_training_data

Returns training data in a suitable format to train a Gaussian process emulator of the type
[`GPmodel](@ref). The training data are (parameter vector, distance) pairs where distances
are the distance from simulated model output to the observed data in summary statistic space.

# Fields
- `n_design_points::Int64`: The number of parameter vectors used to train the Gaussian process emulator
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `distance_metric::Function`: Any function that computes the distance between 2 1D Arrays.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
"""
function get_training_data{D<:ContinuousUnivariateDistribution}(n_design_points::Int64,
        priors::AbstractArray{D,1},
        simulator_function::Function,
        summary_statistic::Union{String,AbstractArray{String,1},Function},
        distance_metric::Function,
        reference_data::AbstractArray{Float64,2})
    X = sample_from_priors(n_design_points, priors)
    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    y = simulate_distance(X, simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
    return X, y
end
