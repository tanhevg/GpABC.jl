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


function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, training_settings::AbstractEmulatorTrainingSettings) where {T<:Real}
    throw("train_emulator(...$(emulation_type)) not implemented")
end

function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, training_settings::DefaultEmulatorTraining) where {T<:Real}
    gpem = GPModel(training_x=training_x, training_y=y, kernel=training_settings.kernel)
    gp_train(gpem)
    gpem
end
