# not exported
function simulate_distance(parameters::AbstractArray{Float64, 2},
        distance_simulation_input::DistanceSimulationInput)
    n_design_points = size(parameters, 1)
    y = zeros(n_design_points)
    for i in 1:n_design_points
        model_output = distance_simulation_input.simulator_function(parameters[i,:])
        y[i] = distance_simulation_input.distance_metric(distance_simulation_input.summary_statistic(model_output),
        distance_simulation_input.reference_summary_statistic)
    end
    y
end

# not exported
# function abc_train_emulator_bak(
#         prior_sampling_function::Function,
#         n_design_points::Int64,
#         reference_summary_statistic,
#         simulator_function::Function,
#         summary_statistic::Function,
#         distance_metric::Function,
#         emulation_type::AbstractEmulatorTrainingSettings = DefaultEmulatorTraining(),
#         repetitive_training::RepetitiveTraining = RepetitiveTraining())
#     X = prior_sampling_function(n_design_points)
#     y = simulate_distance(X, simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
#     gpem = train_emulator(X, reshape(y, (length(y), 1)), emulation_type)
#     # gpem = GPModel(training_x=X, training_y=y, kernel=gpkernel)
#     # gp_train(gpem)
#     for rt_count in 1:repetitive_training.rt_iterations
#         retraining_sample = prior_sampling_function(repetitive_training.rt_sample_size)
#         mean, variance = gp_regression(retraining_sample, gpem)
#         variance_perm = sortperm(variance, rev=true)
#         idx = variance_perm[1:repetitive_training.rt_extra_training_points]
#         extra_x = retraining_sample[idx, :]
#         extra_y = simulate_distance(extra_x,
#             simulator_function, summary_statistic, distance_metric, reference_summary_statistic)
#         new_training_x = vcat(gpem.gp_training_x, extra_x)
#         new_training_y = vcat(gpem.gp_training_y, extra_y)
#         info("Iteration $(rt_count). ",
#             "Adding $(length(extra_y)) training points. ",
#             "Variances: $(variance[idx]). ",
#             "Distances: $(extra_y). ",
#             "Total number of training points: $(length(new_training_y))"; prefix="GpABC Repetitive training ")
#         gpem = train_emulator(new_training_x, new_training_y, emulation_type)
#         # gpem = GPModel(training_x=new_training_x, training_y=new_training_y, kernel=gpem.kernel)
#         # gp_train(gpem)
#     end
#     gpem
# end

function abc_train_emulator(
    prior_sampling_function::Function,
    n_design_points::Int64,
    training_input::EmulatorTrainingInput
    )
    X = prior_sampling_function(n_design_points)
    y = simulate_distance(X, training_input.distance_simulation_input)
    train_emulator(X, reshape(y, (length(y), 1)), training_input.emulator_training)
end

function abc_retrain_emulator(
    gpem::GPModel,
    prior_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::AbstractRetrainingSettings
    ) where {T<:Real}
    throw("abc_retrain_emulator(...$(typeof(retraining_settings))) not implemented")
end

function abc_retrain_emulator(
    gpm::GPModel,
    particle_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::DefaultRetrainingSettings
    ) where {T<:Real}
    n_accepted = 0
    n_simulations = 0
    extra_x = zeros(retraining_settings.design_points, size(gpm.gp_training_x, 2))
    extra_y = zeros(retraining_settings.design_points, 1)
    while n_accepted < retraining_settings.design_points && n_simulations < retraining_settings.max_simulations
        particle = particle_sampling_function(1)
        distance = simulate_distance(particle, training_input.distance_simulation_input)
        # println("particle: $(particle)")
        # println("distance: $(distance)")
        if distance[1] < epsilon
            n_accepted += 1
            extra_x[n_accepted, :] = particle[1]
            extra_y[n_accepted] = distance[1]
        end
        n_simulations += 1
    end
    if n_accepted > 0
        training_x = vcat(gpm.gp_training_x, extra_x[1:n_accepted, :])
        training_y = vcat(gpm.gp_training_y, extra_y[1:n_accepted])
        info("Accepted $(n_accepted) new training points below threshold $(epsilon) after $(n_simulations) simulations"; prefix="GpABC Emulator retraining ")
        train_emulator(training_x, training_y, training_input.emulator_training)
    else
        warn("No new training points accepted below threshold $(epsilon); the emulator is not changed"; prefix="GpABC Emulator retraining ")
        gpm
    end
end

function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, et::AbstractEmulatorTraining) where {T<:Real}
    throw("train_emulator(...$(typeof(et))) not implemented")
end

function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, emulator_training::DefaultEmulatorTraining) where {T<:Real}
    gpem = GPModel(training_x=training_x, training_y=y, kernel=emulator_training.kernel)
    gp_train(gpem)
    gpem
end
