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
        distance_metric::Function;
        gpkernel::AbstractGPKernel=SquaredExponentialArdKernel(),
        rts::RepetitiveTraining = RepetitiveTraining())
    X = prior_sampling_function(n_design_points)
    y = simulate_distance(X,
        simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
    gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
    gp_train(gpem)
    rt_count = 0
    while rt_count < rts.rt_iterations
        retraining_sample = prior_sampling_function(rts.rt_sample_size)
        mean, variance = gp_regression(retraining_sample, gpem)
        variance_perm = sortperm(variance, rev=true)
        idx = variance_perm[1:rts.rt_extra_training_points]
        extra_x = retraining_sample[idx, :]
        extra_y = simulate_distance(extra_x,
            simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
        # println("Repetitive training. Adding extra $(rts.rt_extra_training_points) training points.")
        # println("X = $extra_x")
        # println("y = $extra_y")
        # println("variance = $(variance[idx])")
        new_training_x = vcat(gpem.gp_training_x, extra_x)
        new_training_y = vcat(gpem.gp_training_y, extra_y)
        gpem = GPModel(training_x=new_training_x, training_y=new_training_y, kernel=gpem.kernel)
        gp_train(gpem)
        rt_count += 1
    end
    gpem
end
