# not exported
function simulate_distance(parameters::AbstractArray{Float64, 2},
        distance_simulation_input::DistanceSimulationInput)
    n_design_points = size(parameters, 1)
    y = zeros(n_design_points)
    for i in 1:n_design_points
        model_output = distance_simulation_input.simulator_function(parameters[i,:])
        y[i] = distance_simulation_input.distance_metric(
            distance_simulation_input.summary_statistic(model_output),
            distance_simulation_input.reference_summary_statistic
        )
    end
    y
end

function abc_train_emulator(priors::AbstractArray{CUD}, training_input::EmulatorTrainingInput) where {CUD<:ContinuousUnivariateDistribution}
    n_dims = length(priors)
    priors = reshape(priors, 1, n_dims)
    X = zeros(training_input.design_points, n_dims)
    X .= rand.(priors)
    y = simulate_distance(X, training_input.distance_simulation_input)
    train_emulator(X, reshape(y, (length(y), 1)), training_input.emulator_training)
end

"""
    abc_retrain_emulator(
        gp_model,
        particle_sampling_function,
        epsilon,
        training_input::EmulatorTrainingInput,
        retraining_settings<:AbstractEmulatorRetraining
        )

Additional retraining procedure for the emulator that may or may not be executed before every iteration of emulation based ABC. Details of the procedure are determined by the subtype of [`AbstractEmulatorRetraining`](@ref) that is passed as an argument.
"""
function abc_retrain_emulator end

function abc_retrain_emulator(
    gpm::GPModel,
    particle_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::IncrementalRetraining
    ) where {T<:Real}
    n_accepted = 0
    n_simulations = 0
    extra_x = zeros(retraining_settings.design_points, size(gpm.gp_training_x, 2))
    extra_y = zeros(retraining_settings.design_points, 1)
    while n_accepted < retraining_settings.design_points && n_simulations < retraining_settings.max_simulations
        particle = particle_sampling_function(1)
        distance = simulate_distance(particle, training_input.distance_simulation_input)
        if distance[1] < epsilon
            n_accepted += 1
            extra_x[n_accepted, :] = particle[1, :]
            extra_y[n_accepted] = distance[1]
        end
        n_simulations += 1
    end
    if n_accepted > 0
        training_x = vcat(gpm.gp_training_x, extra_x[1:n_accepted, :])
        training_y = vcat(gpm.gp_training_y, extra_y[1:n_accepted])
        @info "GpABC Emulator retraining. Accepted $(n_accepted) new training points below threshold $(epsilon) after $(n_simulations) simulations"
        train_emulator(training_x, training_y, training_input.emulator_training)
    else
        @warn "No new training points accepted below threshold $(epsilon); the emulator is not changed"; prefix="GpABC Emulator retraining "
        gpm
    end
end

function abc_retrain_emulator(
    gpm::GPModel,
    particle_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::PreviousPopulationRetraining
    ) where {T<:Real}
    n_design_points = size(gpm.gp_training_x, 1)
    training_x = particle_sampling_function(n_design_points)
    training_y = simulate_distance(training_x, training_input.distance_simulation_input)
    train_emulator(training_x, reshape(training_y, n_design_points, 1), training_input.emulator_training)
end


function abc_retrain_emulator(
    gpm::GPModel,
    particle_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::NoopRetraining
    ) where {T<:Real}
    gpm
end

function abc_retrain_emulator(
    gpm::GPModel,
    particle_sampling_function::Function,
    epsilon::T,
    training_input::EmulatorTrainingInput,
    retraining_settings::PreviousPopulationThresholdRetraining
    ) where {T<:Real}
    cap_below_threshold = retraining_settings.n_design_points
    cap_above_threshold = retraining_settings.n_design_points
    training_x_below_threshold = zeros(cap_below_threshold, size(gpm.gp_training_x, 2))
    training_x_above_threshold = zeros(cap_above_threshold, size(gpm.gp_training_x, 2))
    training_y_below_threshold = zeros(cap_below_threshold, 1)
    training_y_above_threshold = zeros(cap_above_threshold, 1)
    n_below_threshold = 0
    n_above_threshold = 0
    n_iter = 0
    while(n_below_threshold < retraining_settings.n_below_threshold && n_iter < retraining_settings.max_iter)
        sample_x = particle_sampling_function(retraining_settings.n_design_points)
        sample_y = simulate_distance(sample_x, training_input.distance_simulation_input)
        idx_below_threshold = findall(sample_y .<= epsilon)
        idx_above_threshold = filter(x->!(x in idx_below_threshold), axes(sample_y, 1))
        @info "GpABC Emulator retraining. Iteration $(n_iter + 1): $(length(idx_below_threshold)) design points with distance below $(epsilon)"
        if n_below_threshold + length(idx_below_threshold) > cap_below_threshold
            idx_below_threshold = idx_below_threshold[1:cap_below_threshold - n_below_threshold]
            idx_above_threshold = vcat(idx_below_threshold[cap_below_threshold - n_below_threshold + 1:end], idx_above_threshold, )
        end
        if n_above_threshold + length(idx_above_threshold) > cap_above_threshold
            idx_above_threshold = idx_above_threshold[1:cap_above_threshold - n_above_threshold]
        end
        end_below_threshold = n_below_threshold + length(idx_below_threshold)
        end_above_threshold = n_above_threshold + length(idx_above_threshold)
        if end_below_threshold > n_below_threshold
            training_x_below_threshold[n_below_threshold + 1:end_below_threshold,:] .= sample_x[idx_below_threshold, :]
            training_y_below_threshold[n_below_threshold + 1:end_below_threshold] .= sample_y[idx_below_threshold]
        end
        if end_above_threshold > n_above_threshold
            training_x_above_threshold[n_above_threshold + 1:end_above_threshold,:] .= sample_x[idx_above_threshold, :]
            training_y_above_threshold[n_above_threshold + 1:end_above_threshold] .= sample_y[idx_above_threshold]
        end
        n_below_threshold = end_below_threshold
        n_above_threshold = end_above_threshold
        n_iter += 1
    end
    if n_above_threshold == 0
        @warn "No design points with distance below $(epsilon) were accepted"
    end
    n_above_threshold = min(n_above_threshold, retraining_settings.n_design_points - n_below_threshold)
    training_x = vcat(training_x_below_threshold[1:n_below_threshold, :], training_x_above_threshold[1:n_above_threshold, :])
    training_y = vcat(training_y_below_threshold[1:n_below_threshold, :], training_y_above_threshold[1:n_above_threshold, :])
    train_emulator(training_x, training_y, training_input.emulator_training)
end

"""
    train_emulator(training_x, training_y, emulator_training<:AbstractEmulatorTraining)

Train the emulator. For custom training procedure, a new subtype of [`AbstractEmulatorTraining`](@ref)
should be created, and a method of this function should be created for it.

# Returns
A trained emulator. Shipped implementations return a [`GPModel`](@ref).
"""
function train_emulator end

function train_emulator(training_x::AbstractArray{T, 2}, y::AbstractArray{T, 2}, emulator_training::DefaultEmulatorTraining) where {T<:Real}
    gpem = GPModel(training_x=training_x, training_y=y, kernel=emulator_training.kernel)
    gp_train(gpem)
    gpem
end

"""
    abc_select_emulated_particles(emulator, particles, threshold, selection<:AbstractEmulatedParticleSelection)

Compute the approximate distances by running the regression-based `emulator` on the provided `particles`, and return the accepted particles. Acceptance strategy is determined by `selection` - subtype of [`AbstractEmulatedParticleSelection`](@ref).

# Arguments
- `gpm`: the [`GPModel`](@ref), that contains the training data (x and y),
  the kernel, the hyperparameters and the test data for running the regression.
- `parameters`: array of parameters to test (the particles).
- `threshold`: if the distance for a particle is below `threshold` then it is accepted as a posterior sample.
- `selection`: the acceptance strategy (a subtype of [`AbstractEmulatedParticleSelection`](@ref)). This determines the method by which an emulated distance is accepted.

# Return
A tuple of accepted distances, and indices of the accepted particles in the supplied `particles` array.
"""
function abc_select_emulated_particles end

function abc_select_emulated_particles(gpm::GPModel, parameters::AbstractArray{T, 2},
        threshold::T, selection::MeanEmulatedParticleSelection) where {T<:Real}
    distances, vars = gp_regression(parameters, gpm)
    accepted_indices = findall(distances .<= threshold)
    distances[accepted_indices], accepted_indices
end

function abc_select_emulated_particles(gpm::GPModel, parameters::AbstractArray{T, 2},
        threshold::T, selection::MeanVarEmulatedParticleSelection) where {T<:Real}
    distances, vars = gp_regression(parameters, gpm)
    accepted_indices = findall((distances .<= threshold) .& (sqrt.(vars) .<= selection.variance_threshold_factor * threshold))
    distances[accepted_indices], accepted_indices
end

function abc_select_emulated_particles(gpm::GPModel, parameters::AbstractArray{T, 2},
        threshold::T, selection::PosteriorSampledEmulatedParticleSelection) where {T<:Real}
    distances = gp_regression_sample(parameters, 1, gpm)
    accepted_indices = findall(distances .<= threshold)
    distances[accepted_indices], accepted_indices
end
