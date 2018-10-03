# not exported
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

# not exported
function abc_train_emulator(
        prior_sampling_function::Function,
        n_design_points::Int64,
        reference_summary_statistic,
        simulator_function::Function,
        summary_statistic::Function,
        distance_metric::Function,
        emulation_type::AbstractEmulatorTrainingSettings = DefaultEmulatorTraining(),
        repetitive_training::RepetitiveTraining = RepetitiveTraining())
    X = prior_sampling_function(n_design_points)
    y = simulate_distance(X, simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
    gpem = train_emulator(X, reshape(y, (length(y), 1)), emulation_type)
    # gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    # gp_train(gpem)
    for rt_count in 1:repetitive_training.rt_iterations
        retraining_sample = prior_sampling_function(repetitive_training.rt_sample_size)
        mean, variance = gp_regression(retraining_sample, gpem)
        variance_perm = sortperm(variance, rev=true)
        idx = variance_perm[1:repetitive_training.rt_extra_training_points]
        extra_x = retraining_sample[idx, :]
        extra_y = simulate_distance(extra_x,
            simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
        new_training_x = vcat(gpem.gp_training_x, extra_x)
        new_training_y = vcat(gpem.gp_training_y, extra_y)
        info("Iteration $(rt_count). ",
            "Adding $(length(extra_y)) training points. ",
            "Variances: $(variance[idx]). ",
            "Distances: $(extra_y). ",
            "Total number of training points: $(length(new_training_y))"; prefix="GpABC Repetitive training ")
        gpem = train_emulator(new_training_x, new_training_y, emulation_type)
        # gpem = GPModel(training_x=new_training_x, training_y=new_training_y, kernel=gpem.kernel)
        # gp_train(gpem)
    end
    gpem
end

"""
    get_training_data{D<:ContinuousUnivariateDistribution}(input::LNAInput,
        n_samples::Int64, n_design_points::Int64,
        priors::AbstractArray{D,1},
        summary_statistic::Union{String,AbstractArray{String,1},Function},
        distance_metric::Function,
        reference_data::AbstractArray{Float64,2},
        x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
        Tspan::Tuple{Float64,Float64},
        saveat::Float64,
        unknown_param_idxs::Union{AbstractArray{Int64,1},Void}=nothing,
        solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();
        kwargs...)

This is the same as `get_training_data` but can compute the LNA.
Returns training data in a suitable format to train a Gaussian process emulator of the type
[`GPmodel](@ref). The training data are (parameter vector, distance) pairs where distances
are the distance from simulated model output to the observed data in summary statistic space.

# Fields
- `input::LNAInput`: LNAInput structure - fields described in lna.jl
- `n_design_points::Int64`: The number of parameter vectors used to train the Gaussian process emulator
- `priors::AbstractArray{D,1}`: A 1D Array of continuous univariate distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `distance_metric::Function`: Any function that computes the distance between 2 1D Arrays.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `x0::Tuple{AbstractArray{Float64,2},AbstractArray{Float64,2}}`: The initial conditions of the system. In the form of (the initial conditions of the species, the initial covariance matrix of the system).
- `Tspan::Tuple{Float64,Float64}`: The start and end times of the simulation
- `saveat::Float64`: The number of time points the use wishes to solve the system for
- `unknown_param_idxs::Union{AbstractArray{Int64,1},Void}=nothing`: The indices of the parameters that are unknown and the user wishes to estimate. If is argument is not supplied it is assumed the first n parameters are unknown, where n is the number of priors provided.
- `solver::DEAlgorithm`: The ODE solver the user wishes to use, for example DifferentialEquations.RK4()
"""

function get_training_data{D<:ContinuousUnivariateDistribution}(input::LNAInput,
    n_samples::Int64, n_design_points::Int64,
    priors::AbstractArray{D,1},
    summary_statistic::Union{String,AbstractArray{String,1},Function},
    distance_metric::Function,
    reference_data::AbstractArray{Float64,2},
    x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
    Tspan::Tuple{Float64,Float64},
    saveat::Float64,
    unknown_param_idxs::Union{AbstractArray{Int64,1},Void}=nothing,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();
    kwargs...)

    if unknown_param_idxs != nothing
        if length(unknown_param_idxs) != length(priors)
            throw(ArgumentError("Number of unknown parameters does not equal the number of priors"))
        end
    else
        unknown_param_idxs = 1:length(priors)
    end

    X = repmat(input.params', n_design_points, 1)
    y = zeros(n_design_points)

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)

    for (j, idx) in enumerate(unknown_param_idxs)
        X[:,idx] = rand(priors[j], n_design_points)
    end

    for i in 1:n_design_points
        new_input = LNAInput(X[i,:], input.S, input.reaction_rate_function, input.volume)
        model_output = get_LNA_trajectories(new_input, n_samples, x0, Tspan, saveat, solver; kwargs...)
        y[i] = distance_metric(summary_statistic(model_output), reference_summary_statistic)
    end

    return X[:,unknown_param_idxs], y

end

# function build_simulator_function(ode_function::Function,
#     Tspan::Tuple{Float64,Float64}, x0::AbstractArray{Float64,1};
#     solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4(), kwargs...)


function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, training_settings::AbstractEmulatorTrainingSettings) where {T<:Real}
    throw("train_emulator(...$(emulation_type)) not implemented")
end

function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, training_settings::DefaultEmulatorTraining) where {T<:Real}
    gpem = GPModel(training_x=training_x, training_y=y, kernel=training_settings.kernel)
    gp_train(gpem)
    gpem
end
