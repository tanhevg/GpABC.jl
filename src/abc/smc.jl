#
# Returns a 2D array of distributions. The ij-th element is for the
# j-th element of the i-th particle from the previous population
#
function generate_kernels(
        population::AbstractArray{F,2},
        priors::AbstractArray{D,1},
        ) where {
        F<:AbstractFloat,
        D<:ContinuousUnivariateDistribution,
        }
    n_particles = size(population, 1)
    n_params = size(population, 2)

    if n_particles > 1
        stds = std(population, 1)[:]
    else
        stds = 1e-3 * ones(n_params) # If there is only one particle we cannot compute the sd - use a small value instead?
    end

    lowers = minimum.(priors)
    uppers = maximum.(priors)

    CUD = ContinuousUnivariateDistribution
    kernels = Matrix{CUD}(n_particles, n_params)
    for j in 1:n_params
        means = population[:, j]
        kernels[:, j] = TruncatedNormal.(means, stds[j]*sqrt(2.0), lowers[j], uppers[j])
    end

    return kernels
end


function generate_parameters(
        batch_size::Int64,
        priors::AbstractArray{D1,1},
        old_weights::StatsBase.Weights,
        kernels::AbstractArray{D2,2},
        ) where {
        D1, D2<:ContinuousUnivariateDistribution
        }

    n_params = length(priors)

    # ADD DimensionMismatch THROWS SO @inbounds CAN BE USED?

    # the kernels must be centered around the old particles
    # and truncated to the priors.

    particles = StatsBase.sample(indices(kernels, 1), old_weights, batch_size)
    # println("particle: $particles")
    perturbed_parameters = rand.(kernels[particles,:])
    # println("perturbed_parameters = $perturbed_parameters")

    # gives a batch_size x n_params matrix of prior pdfs in perturbed parameters
    pdfs = pdf.(reshape(priors, 1, n_params), perturbed_parameters)
    numerators = prod(pdfs, 2) # multiply across rows, to get a column vector of products of size batch_size

    denominators = zeros(length(numerators), 1)
    for k in eachindex(denominators)
        denominator_pdfs = pdf.(kernels, reshape(perturbed_parameters[k, :], 1, n_params))
        denominator_summands = prod(denominator_pdfs, 2)
        denominators[k] = sum(old_weights .* denominator_summands)
    end

    weight = numerators ./ denominators

    return perturbed_parameters, weight
end

function generate_parameters_no_weights(
        n_batch_size::Int64,
        old_parameters::AbstractArray{F,2},
        old_weights::StatsBase.Weights,
        kernels::AbstractArray{D2,2}
        ) where {
        D2<:ContinuousUnivariateDistribution,
        F<:AbstractFloat,
        }
    particles = StatsBase.sample(indices(old_parameters, 1), old_weights, n_batch_size)
    return rand.(kernels[particles,:])
end

function normalise(
        weights::StatsBase.AbstractWeights;
        tosum = 1.0,
        )
    WeightType = typeof(weights)
    weights = WeightType(weights.values .* (tosum / sum(weights.values)), tosum)

    return weights
end

function normalise(weight_values::AbstractArray{F, 1}, tosum = 1.0) where F<:AbstractFloat
    return StatsBase.Weights(weight_values .* (sum(weight_values) / tosum), tosum)
end

#
# Initialise a simulated ABC-SMC run
#
function initialiseABCSMC(input::SimulatedABCSMCInput,
        reference_data::AbstractArray{Float64,2};
        write_progress = true,
        progress_every = 1000,
        )
    # construct summary statistic function to be used in all runs
    built_summary_statistic = build_summary_statistic(input.summary_statistic)

    # the first run is an ABC rejection simulation
    rejection_input = SimulatedABCRejectionInput(input.n_params,
                                        input.n_particles,
                                        input.threshold_schedule[1],
                                        input.priors,
                                        built_summary_statistic,
                                        input.distance_function,
                                        input.simulator_function,
                                        input.max_iter,
                                        )

    rejection_output = ABCrejection(rejection_input,
                                    reference_data;
                                    write_progress = write_progress,
                                    progress_every = progress_every,
                                    )

    tracker =  SimulatedABCSMCTracker(input.n_params,
                             [rejection_output.n_accepted],
                             [rejection_output.n_tries],
                             [rejection_output.threshold],
                             [rejection_output.population],
                             [rejection_output.distances],
                             [rejection_output.weights],
                             input.priors,
                             built_summary_statistic,
                             input.distance_function,
                             input.simulator_function,
                             input.max_iter,
                             )

    return tracker
end

#
# Initialise an emulated ABC-SMC run
#
function initialiseABCSMC(input::EmulatedABCSMCInput,
        reference_data::AbstractArray{Float64,2};
        write_progress = true)

    # the first run is an ABC rejection simulation
    rejection_input = EmulatedABCRejectionInput(input.n_params,
                                        input.n_particles,
                                        input.threshold_schedule[1],
                                        input.priors,
                                        input.batch_size,
                                        input.max_iter,
                                        input.emulator_training_input)

    rejection_output = ABCrejection(rejection_input,
                                    reference_data;
                                    write_progress = write_progress)

    tracker = EmulatedABCSMCTracker(input.n_params,
                             [rejection_output.n_accepted],
                             [rejection_output.n_tries],
                             [rejection_output.threshold],
                             [rejection_output.population],
                             [rejection_output.distances],
                             [rejection_output.weights],
                             input.priors,
                             input.emulator_training_input,
                             input.emulator_retraining_settings,
                             input.batch_size,
                             input.max_iter,
                             [rejection_output.emulator] # emulators
                             )

    return tracker
end

#
# Iterate a simulated ABC-SMC
#
function iterateABCSMC!(tracker::SimulatedABCSMCTracker,
        threshold::AbstractFloat,
        n_toaccept::Integer,
        reference_data::AbstractArray{Float64,2};
        write_progress = true,
        progress_every = 1000,
        normalise_weights::Bool = true,
        for_model_selection::Bool = false
        )
    if write_progress && !for_model_selection
        info(string(DateTime(now())), " ϵ = $threshold.", prefix="GpABC SMC Simulation ")
    end
    if threshold > tracker.threshold_schedule[end]
        warn("current threshold less strict than previous one.")
    end
    kernels = generate_kernels(tracker.population[end], tracker.priors)
    reference_data_sum_stat = tracker.summary_statistic(reference_data)
    n_tries = 0
    n_accepted = 0
    population = zeros(n_toaccept, tracker.n_params)
    distances = zeros(n_toaccept)
    weights = zeros(n_toaccept)

    # simulate
    while n_accepted < n_toaccept && n_tries < tracker.max_iter
        parameters, weight = generate_parameters(1, tracker.priors, tracker.weights[end], kernels)
        parameters = parameters[1, :]
        weight = weight[1]
        simulated_data = tracker.simulator_function(parameters)
        simulated_data_sum_stat = tracker.summary_statistic(simulated_data)
        # This prevents the whole code from failing if there is a problem with solving the
        # differential equation(s)
        if size(simulated_data_sum_stat) != size(reference_data_sum_stat)
            warn("Summarised simulated and reference data do not have the same size ( $(size(simulated_data_sum_stat)) and $(size(reference_data_sum_stat)) ).
                This may be due to the behaviour of DifferentialEquations::solve - please check for dt-related warnings. Continuing to the next iteration.")
            continue
        end
        distance = tracker.distance_function(reference_data_sum_stat, simulated_data_sum_stat)
        n_tries += 1

        if distance <= threshold
            n_accepted += 1
            population[n_accepted,:] = parameters
            distances[n_accepted] = distance
            weights[n_accepted] = weight
        end

        if write_progress && (n_tries % progress_every == 0)
            info(string(DateTime(now())), " Accepted $(n_accepted)/$(n_tries) particles.", prefix="GpABC SMC Simulation ")
        end
    end

    if n_accepted == 0
        if !for_model_selection
            warn("Simulation reached maximum $(tracker.max_iter) iterations without selecting any particles")
        end
        return false
    end

    if n_accepted < n_toaccept
        population = population[1:n_accepted, :]
        weights = weights[1:n_accepted]
        distances = distances[1:n_accepted]
        if !for_model_selection
            warn("Simulation reached maximum $(tracker.max_iter) iterations before finding $(n_toaccept) particles - will return $n_accepted")
        end
    else
        if !for_model_selection
            info(string(DateTime(now())), " Finished. Accepted $(n_accepted)/$(n_toaccept).", prefix="GpABC SMC Simulation ")
        end
    end
    push!(tracker.n_accepted, n_accepted)
    push!(tracker.n_tries, n_tries)
    push!(tracker.threshold_schedule, threshold)
    push!(tracker.population, population)
    push!(tracker.distances, distances)
    # Do not want to normalise weights now if doing model selection - will do at
    # end of population at model selection level
    if normalise_weights
        weights = normalise(weights)
    end
    push!(tracker.weights, StatsBase.Weights(weights))

    return true
end

#
# Iterate a emulated ABC-SMC
#
function iterateABCSMC!(tracker::EmulatedABCSMCTracker,
        threshold::AbstractFloat,
        n_toaccept::Integer,
        reference_data::AbstractArray{Float64,2};
        write_progress = true,
        progress_every = 1000,
        emulator::Union{GPModel,Void}=nothing,
        normalise_weights::Bool = true,
        for_model_selection::Bool = false
        )
    if write_progress && !for_model_selection
        info(string(DateTime(now())), " ϵ = $threshold.", prefix="GpABC SMC Emulation ")
    end

    old_population = tracker.population[end]
    old_weights = tracker.weights[end]

    if threshold > tracker.threshold_schedule[end]
        println("Warning: current threshold less strict than previous one.")
    end

    kernels = generate_kernels(old_population, tracker.priors)
    if emulator == nothing
        particle_sampling_function(n_design_points) = generate_parameters_no_weights(n_design_points,
            old_population,
            old_weights,
            kernels)

        emulator = abc_retrain_emulator(tracker.emulators[end], particle_sampling_function, threshold,
            tracker.emulator_training_input, tracker.emulator_retraining_settings)
    end

    # initialise
    iter_no = 0
    n_accepted = 0
    all_weights = zeros(n_toaccept)
    population = zeros(n_toaccept, tracker.n_params)
    all_distances = zeros(n_toaccept)
    n_tries = 0

    # emulate
    while n_accepted < n_toaccept && iter_no < tracker.max_iter
        parameters, weights = generate_parameters(tracker.batch_size,
                                                 tracker.priors,
                                                 old_weights,
                                                 kernels)

        distances, vars = gp_regression(parameters, emulator)
        n_tries += length(distances)
        # accepted_indices = find((distances .<= threshold) .& (sqrt.(vars) .<= 0.05 * threshold))
        accepted_indices = find((distances .<= threshold) .& (sqrt.(vars) .<= threshold)) # todo more variance controls
        # accepted_indices = find(distances .<= threshold)
        n_include = length(accepted_indices)
        if n_accepted + n_include > n_toaccept
            n_include = n_toaccept - n_accepted
            accepted_indices = accepted_indices[1:n_include]
        end
        distances = distances[accepted_indices]
        weights = weights[accepted_indices]
        parameters = parameters[accepted_indices, :]
        store_slice = n_accepted + 1 : n_accepted + n_include
        n_accepted += n_include
        population[store_slice,:] = parameters
        all_distances[store_slice] = distances
        all_weights[store_slice] = weights

        if write_progress && !for_model_selection
            info(string(DateTime(now())), " Accepted $(n_accepted)/$(n_tries) particles.", prefix="GpABC SMC Emulation ")
        end

        iter_no += 1

    end

    if n_accepted == 0
        if !for_model_selection
            warn("Emulation reached maximum $(tracker.max_iter) iterations without selecting any particles")
        end
        return false
    end

    if n_accepted < n_toaccept
        population = population[1:n_accepted, :]
        all_weights = all_weights[1:n_accepted]
        all_distances = all_distances[1:n_accepted]
        if !for_model_selection
            warn("Emulation reached maximum $(tracker.max_iter) iterations before finding $(n_toaccept) particles - will return $n_accepted")
        end
    end
    push!(tracker.n_accepted, n_accepted)
    push!(tracker.n_tries, n_tries)
    push!(tracker.threshold_schedule, threshold)
    push!(tracker.population, population)
    push!(tracker.distances, all_distances)
    if normalise_weights
        all_weights = normalise(all_weights)
    else
        all_weights = StatsBase.Weights(all_weights)
    end
    push!(tracker.weights, all_weights)
    push!(tracker.emulators, emulator)

    return true
end

function buildAbcSmcOutput(input::EmulatedABCSMCInput, tracker::EmulatedABCSMCTracker)
    EmulatedABCSMCOutput(input.n_params,
                        tracker.n_accepted,
                        tracker.n_tries,
                        tracker.threshold_schedule,
                        tracker.population,
                        tracker.distances,
                        tracker.weights,
                        tracker.emulators)
end

function buildAbcSmcOutput(input::SimulatedABCSMCInput, tracker::SimulatedABCSMCTracker)
    SimulatedABCSMCOutput(input.n_params,
                        tracker.n_accepted,
                        tracker.n_tries,
                        tracker.threshold_schedule,
                        tracker.population,
                        tracker.distances,
                        tracker.weights)
end

"""
  ABCSMC

Run a ABC-SMC computation using either simulation (the model is simulated in full for each parameter vector from which the corresponding
distance to observed data is used to construct the posterior) or emulation (a regression model trained to predict the distance from the
parameter vector directly is used to construct the posterior). Whether simulation or emulation is used is controlled by the type of `input`.

# Arguments
- `input::ABCSMCInput`: An ['SimulatedABCSMCInput'](@ref) or ['EmulatedABCSMCInput'](@ref) object that defines the settings for the ABC-SMC run.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `write_progress::Bool`: Optional argument controlling whether progress is written to `out_stream`.
- `progress_every::Int`: Progress will be written to `out_stream` every `progress_every` simulations (optional, ignored if `write_progress` is `False`).

# Return
An object that inherits from ['ABCSMCOutput'](@ref), depending on whether a `input` is a ['SimulatedABCSMCInput'](@ref) or ['EmulatedABCSMCInput'](@ref).
"""
function ABCSMC(
        input::ABCSMCInput,
        reference_data::AbstractArray{Float64,2};
        write_progress = true,
        progress_every = 1000,
        )
    n_toaccept = input.n_particles

    tracker = initialiseABCSMC(input,
                               reference_data;
                               write_progress = write_progress)

    if tracker.n_accepted[1] > 0
        for i in 2:length(input.threshold_schedule)
            # @assert size(tracker.population[end], 1) > 0 "No particles were accepted by step #$(i-1) of ABC SMC"
            threshold = input.threshold_schedule[i]
            complete_threshold = iterateABCSMC!(tracker,
                           threshold,
                           input.n_particles,
                           reference_data;
                           write_progress = write_progress,
                           progress_every = progress_every,
                           )
            if !complete_threshold
                break
            end
        end
    else
        warn("No particles selected at initial rejection ABC step of SMC ABC")
    end

    return buildAbcSmcOutput(input, tracker)
end
