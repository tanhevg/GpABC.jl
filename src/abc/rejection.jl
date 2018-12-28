#
# Returns a single parameter and its weight - for simulation
#
function generate_parameters(
        priors::AbstractArray{D,1},
        ) where {
        D<:ContinuousUnivariateDistribution
        }
    n_dims = length(priors)
    parameters = zeros(n_dims)

    weight = 1.
    @inbounds for i in 1:n_dims
        parameters[i] = rand(priors[i])
        weight *= pdf(priors[i], parameters[i])
    end

    return reshape(parameters, (1, n_dims)), weight
end

#
# Returns a set parameter and its weight with size batch_size - for emulation
#
function generate_parameters(
        priors::AbstractArray{D,1},
        batch_size::Int,
        ) where {
        D<:ContinuousUnivariateDistribution
        }

    n_dims = length(priors)
    parameters = zeros(batch_size, n_dims)
    priors = reshape(priors, 1, n_dims)
    parameters .= rand.(priors)
    weights = prod(pdf.(priors, parameters), dims=2)
    weights = reshape(weights, batch_size) # So that this returns a 1D array, like the simulation version

    return parameters, weights
end

function ABCrejection(input::SimulatedABCRejectionInput;
    write_progress::Bool = true,
    progress_every::Int = 1000)

	checkABCInput(input)
    if write_progress
        @info "GpABC rejection simulation. ϵ = $(input.threshold)."
    end

	# initialise
    n_tries = 0
    n_accepted = 0
    accepted_parameters = zeros(input.n_particles, input.n_params)
    accepted_distances = zeros(input.n_particles)
    weight_values = zeros(input.n_particles)
    distance = 0.0

    # simulate
    while n_accepted < input.n_particles && n_tries < input.max_iter

        parameters, weight_value = generate_parameters(input.priors)
        try
            distance = simulate_distance(parameters, input.distance_simulation_input)
        catch e
            if isa(e, DimensionMismatch)
                # This prevents the whole code from failing if there is a problem
                # solving the differential equation(s). The exception is thrown by the
                # distance function
                @warn "The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of DifferentialEquations::solve - please check for related warnings. Continuing to the next iteration."
                n_tries += 1
                continue
            else
                throw(e)
            end
        end

        n_tries += 1

        if distance[1] <= input.threshold
            n_accepted += 1
            accepted_parameters[n_accepted,:] = parameters
            accepted_distances[n_accepted] = distance[1]
            weight_values[n_accepted] = weight_value
        end

        if write_progress && (n_tries % progress_every == 0)
            @info "GpABC rejection simulation. Accepted $(n_accepted)/$(n_tries) particles."
        end
    end

    if n_accepted < input.n_particles
        @warn "Simulation reached maximum iterations $(input.max_iter) before finding $(input.n_particles) particles - will return $n_accepted"
        accepted_parameters = accepted_parameters[1:n_accepted, :]
        accepted_distances = accepted_distances[1:n_accepted]
        weight_values = weight_values[1:n_accepted]
    end


    # output
    output = SimulatedABCRejectionOutput(input.n_params,
                                n_accepted,
                                n_tries,
                                input.threshold,
                                accepted_parameters,
                                accepted_distances,
                                StatsBase.Weights(weight_values ./ sum(weight_values)) # normalise weights
                                )

    return output

end

function ABCrejection(input::EmulatedABCRejectionInput;
    write_progress = true,
    progress_every = 1000)

    checkABCInput(input)

    if write_progress
        @info "GpABC rejection emulation. ϵ = $(input.threshold)."
    end
    # initialise
    n_accepted = 0
    n_tries = 0
    batch_no = 1
    accepted_parameters = zeros(input.n_particles, input.n_params)
    accepted_distances = zeros(input.n_particles)
    weights = ones(input.n_particles)

    # emulate
    emulator = abc_train_emulator(input.priors, input.emulator_training_input)
    while n_accepted < input.n_particles && batch_no <= input.max_iter
        parameter_batch, weight_batch = generate_parameters(input.priors, input.batch_size)

        # distances, vars = gp_regression(parameter_batch, emulator)
        n_tries += input.batch_size
        #
        # Check which parameter indices were accepted
        #
        distances, accepted_batch_idxs = abc_select_emulated_particles(emulator, parameter_batch, input.threshold, input.selection)

        n_accepted_batch = length(accepted_batch_idxs)

        #
        # If some parameters were accepted, store their values, (predicted) distances and weights
        #
        if n_accepted_batch > 0
            # Check that we won't accept too many parameters - throw away the extra parameters if so
            if n_accepted + n_accepted_batch > input.n_particles
                accepted_batch_idxs = accepted_batch_idxs[1:input.n_particles - n_accepted]
                n_accepted_batch = length(accepted_batch_idxs)
                distances = distances[1:n_accepted_batch]
            end

            accepted_parameters[n_accepted+1:n_accepted + n_accepted_batch,:] = parameter_batch[accepted_batch_idxs,:]
            accepted_distances[n_accepted+1:n_accepted + n_accepted_batch] = distances
            weights[n_accepted+1:n_accepted + n_accepted_batch] = weight_batch[accepted_batch_idxs]
            n_accepted += n_accepted_batch

        end

        if write_progress
            @info "GpABC rejection emulation. Accepted $(n_accepted)/$(n_tries) particles ($(batch_no) batches of size $(input.batch_size))."
        end

        batch_no += 1
    end

    if n_accepted < input.n_particles
        @warn "Emulation reached maximum $(input.max_iter) iterations before finding $(input.n_particles) particles - will return $n_accepted"
        accepted_parameters = accepted_parameters[1:n_accepted, :]
        accepted_distances = accepted_distances[1:n_accepted]
        weights = weights[1:n_accepted]
    end

    # output
    output = EmulatedABCRejectionOutput(input.n_params,
                                n_accepted,
                                n_tries,
                                input.threshold,
                                accepted_parameters,
                                accepted_distances,
                                StatsBase.Weights(weights ./ sum(weights)),
                                emulator
                                )

    return output
end
