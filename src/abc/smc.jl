#
# Returns a 2D array of distributions. The ij-th element is for the
# j-th element of the i-th particle from the previous population
#
function generate_kernels(
        population::Matrix{F},
        priors::Vector{D},
        ) where {
        F<:AbstractFloat,
        D<:ContinuousUnivariateDistribution,
        }
    n_particles = size(population, 1)
    n_params = size(population, 2)

    stds = std(population, 1)[:]
    lowers = minimum.(priors)
    uppers = maximum.(priors)

    CUD = ContinuousUnivariateDistribution
    kernels = Matrix{CUD}(n_particles, n_params)
    for j in 1:n_params
        means = population[j,:]
        for i in 1:n_particles
            kernels[i, j] = TruncatedNormal(means[j],
                                                          stds[j]*sqrt(2),
                                                          lowers[j],
                                                          uppers[j],
                                                          )
        end
    end

    return kernels
end


function generate_parameters(
        priors::Vector{D1},
        old_parameters::Matrix{F},
        old_weights::StatsBase.Weights,
        kernels::Matrix{D2},
        ) where {
        D1, D2<:ContinuousUnivariateDistribution,
        F<:AbstractFloat,
        }

    n_params = length(priors)

    # ADD DimensionMismatch THROWS SO @inbounds CAN BE USED?

    # the kernels must be centered around the old particles
    # and truncated to the priors.

    particle = StatsBase.sample(indices(old_parameters, 1), old_weights)
    perturbed_parameters = rand.(kernels[particle,:])
    #println("perturbed_parameters = $perturbed_parameters")

    numerator = 1.0
    for i in 1:n_params
        numerator *= pdf(priors[i], perturbed_parameters[i])
    end

    denominator = 0.0
    for k in eachindex(old_weights)
        # calculate the total kernel
        kernel = 1.0
        for j in 1:n_params
            kernel *= pdf(kernels[k,j], perturbed_parameters[j])
        end
        denominator += old_weights[k] * kernel

        # weight normalisation---for numerical stability if nothing else
        denominator *= old_weights.sum
    end

    weight = numerator / denominator

    return perturbed_parameters, weight
end


function normalise(
        weights::StatsBase.AbstractWeights;
        tosum = 1.0,
        )
    WeightType = typeof(weights)
    weights = WeightType(weights.values .* (tosum / sum(weights.values)), tosum)

    return weights
end

#
# Initialise a simulated ABC-SMC run
#
function initialiseABCSMC(
        input::SimulatedABCSMCInput,
        reference_data;
        out_stream::IO =  STDOUT,
        write_progress = true,
        progress_every = 1000,
        )
    # the first run is an ABC rejection simulation
    rejection_input = SimulatedABCRejectionInput(input.n_params,
                                        input.n_particles,
                                        input.threshold_schedule[1],
                                        input.priors,
                                        input.distance_function,
                                        input.data_generating_function,
                                        )

    rejection_output = ABCrejection(rejection_input,
                                    reference_data;
                                    out_stream = out_stream,
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
                             input.distance_function,
                             input.data_generating_function,
                             )

    return tracker
end

#
# Initialise an emulated ABC-SMC run
#
function initialiseABCSMC(
        input::EmulatedABCSMCInput,
        reference_data;
        out_stream::IO =  STDOUT,
        write_progress = true,
        progress_every = 1000,
        )
    # the first run is an ABC rejection simulation
    rejection_input = EmulatedABCRejectionInput(input.n_params,
                                        input.n_particles,
                                        input.threshold_schedule[1],
                                        input.priors,
                                        input.distance_prediction_function,
                                        input.batch_size,
                                        input.max_iter
                                        )

    rejection_output = ABCrejection(rejection_input,
                                    reference_data;
                                    out_stream = out_stream,
                                    write_progress = write_progress,
                                    progress_every = progress_every,
                                    )

    tracker =  EmulatedABCSMCTracker(input.n_params,
                             [rejection_output.n_accepted],
                             [rejection_output.n_tries],
                             [rejection_output.threshold],
                             [rejection_output.population],
                             [rejection_output.distances],
                             [rejection_output.weights],
                             input.priors,
                             input.distance_prediction_function)

    return tracker
end

#
# Iterate a simulated ABC-SMC
#
function iterateABCSMC!(
        tracker::SimulatedABCSMCTracker,
        threshold::AbstractFloat,
        n_toaccept::Integer,
        reference_data;
        out_stream::IO = STDOUT,
        write_progress = true,
        progress_every = 1000,
        )
    # initialise
    push!(tracker.n_accepted, 0)
    push!(tracker.n_tries, 0)
    if threshold > tracker.threshold_schedule[end]
        println("Warning: current threshold less strict than previous one.")
    end
    push!(tracker.threshold_schedule, threshold)
    push!(tracker.population, zeros(tracker.population[end]))
    push!(tracker.distances, zeros(tracker.distances[end]))
    push!(tracker.weights, StatsBase.Weights(ones(tracker.weights[end].values)))

    kernels = generate_kernels(tracker.population[end-1], tracker.priors)

    # simulate
    while tracker.n_accepted[end] < n_toaccept
        parameters, weight = generate_parameters(tracker.priors,
                                                 tracker.population[end-1],
                                                 tracker.weights[end-1],
                                                 kernels,
                                                 )

        simulated_data = tracker.data_generating_function(parameters)
        distance = tracker.distance_function(reference_data, simulated_data)
        tracker.n_tries[end] += 1

        if distance <= threshold
            tracker.n_accepted[end] += 1
            n_accepted = tracker.n_accepted[end]
            tracker.population[end][n_accepted,:] = parameters
            tracker.distances[end][n_accepted] = distance
            tracker.weights[end].values[n_accepted] = weight
        end

        if write_progress && (tracker.n_tries[end] % progress_every == 0)
            write(out_stream, string(DateTime(now())),
                              " Accepted ",
                              string(tracker.n_accepted[end]),
                              "/",
                              string(tracker.n_tries[end]),
                              " particles.\n"
                              )
            flush(out_stream)
        end
    end

    tracker.weights[end] = deepcopy(normalise(tracker.weights[end], tosum=1.0))

    return tracker
end

#
# Iterate a emulated ABC-SMC
#
function iterateABCSMC!(
        tracker::EmulatedABCSMCTracker,
        threshold::AbstractFloat,
        n_toaccept::Integer,
        reference_data;
        out_stream::IO = STDOUT,
        write_progress = true,
        progress_every = 1000,
        )
    # initialise
    push!(tracker.n_accepted, 0)
    push!(tracker.n_tries, 0)
    if threshold > tracker.threshold_schedule[end]
        println("Warning: current threshold less strict than previous one.")
    end
    push!(tracker.threshold_schedule, threshold)
    push!(tracker.population, zeros(tracker.population[end]))
    push!(tracker.distances, zeros(tracker.distances[end]))
    push!(tracker.weights, StatsBase.Weights(ones(tracker.weights[end].values)))

    kernels = generate_kernels(tracker.population[end-1], tracker.priors)

    # emulate
    while tracker.n_accepted[end] < n_toaccept
        parameters, weight = generate_parameters(tracker.priors,
                                                 tracker.population[end-1],
                                                 tracker.weights[end-1],
                                                 kernels,
                                                 )

        #
        # Need to transpose parameter vector to pass it to emulator
        #
        #println("parameters have size $(size(parameters))")
        distance = tracker.distance_prediction_function(parameters')[1]
        #println("distance = $distance")
        tracker.n_tries[end] += 1

        if distance <= threshold
            tracker.n_accepted[end] += 1
            n_accepted = tracker.n_accepted[end]
            tracker.population[end][n_accepted,:] = parameters
            tracker.distances[end][n_accepted] = distance
            tracker.weights[end].values[n_accepted] = weight
        end

        if write_progress && (tracker.n_tries[end] % progress_every == 0)
            write(out_stream, string(DateTime(now())),
                              " Accepted ",
                              string(tracker.n_accepted[end]),
                              "/",
                              string(tracker.n_tries[end]),
                              " particles.\n"
                              )
            flush(out_stream)
        end
    end

    tracker.weights[end] = deepcopy(normalise(tracker.weights[end], tosum=1.0))

    return tracker
end


function ABCSMC(
        input::ABCSMCInput,
        reference_data;
        out_stream::IO = STDOUT,
        write_progress = true,
        progress_every = 1000,
        )
    n_toaccept = input.n_particles

    tracker = initialiseABCSMC(input,
                               reference_data;
                               out_stream = out_stream,
                               write_progress = write_progress,
                               progress_every = progress_every,
                               )

    for i in 2:length(input.threshold_schedule)
        threshold = input.threshold_schedule[i]
        iterateABCSMC!(tracker,
                       threshold,
                       input.n_particles,
                       reference_data;
                       out_stream = out_stream,
                       write_progress = write_progress,
                       progress_every = progress_every,
                       )
        output = ABCSMCOutput(input.n_params,
                              tracker.n_accepted,
                              tracker.n_tries,
                              tracker.threshold_schedule,
                              tracker.population,
                              tracker.distances,
                              tracker.weights,
                              )

        #write(out_stream, output)
    end

    return ABCSMCOutput(input.n_params,
                        tracker.n_accepted,
                        tracker.n_tries,
                        tracker.threshold_schedule,
                        tracker.population,
                        tracker.distances,
                        tracker.weights,
                        )
end
