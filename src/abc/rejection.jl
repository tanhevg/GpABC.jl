#
# Returns a single parameter and its weight - for simulation
#
function generate_parameters(
        priors::Vector{D},
        ) where {
        D<:ContinuousUnivariateDistribution
        }
    n_dims = length(priors)
    parameters = zeros(n_dims)

    weight = 1.
    for i in 1:n_dims
        @inbounds parameters[i] = rand(priors[i])
        weight *= pdf(priors[i], parameters[i])
    end

    return parameters, weight
end

#
# Returns a set parameter and its weight with size batch_size - for emulation
#
function generate_parameters(
        priors::Vector{D},
        batch_size::Int,
        ) where {
        D<:ContinuousUnivariateDistribution
        }
    n_dims = length(priors)
    parameters = zeros(batch_size, n_dims)
    weights = ones(batch_size)

    for j in 1:n_dims
        for i in 1:batch_size
            @inbounds parameters[i,j] = rand(priors[j])
            weights[i] *= pdf(priors[j], parameters[j])
        end
    end

    return parameters, weights
end

#
# Note - have removed simulation_args... should pass an anonymous function that returns
# simulation results with the parameters as the only argument to SimulatedABCRejectionInput
# as data_generating_function
#

#
# Simulated version
#
function ABCrejection(
	input::SimulatedABCRejectionInput,
	reference_data;
	out_stream::IO = STDOUT,
    write_progress = true,
    progress_every = 1000)

	checkABCInput(input)

	# initialise
    n_tries = 0
    n_accepted = 0
    accepted_parameters = zeros(input.n_particles, input.n_params)
    accepted_distances = zeros(input.n_particles)
    weights = ones(input.n_particles)

    # simulate
    while n_accepted < input.n_particles
        parameters, weight = generate_parameters(input.priors)
        simulated_data = input.data_generating_function(parameters)
        distance = input.distance_function(reference_data, simulated_data)
        n_tries += 1

        if distance <= input.threshold
            n_accepted += 1
            accepted_parameters[n_accepted,:] = parameters
            accepted_distances[n_accepted] = distance
            weights[n_accepted] = weight
        end

        if write_progress && (n_tries % progress_every == 0)
            write(out_stream, string(DateTime(now())),
                              " Accepted ",
                              string(n_accepted),
                              "/",
                              string(n_tries),
                              " particles.\n"
                              )
            flush(out_stream)
        end
    end

    weights = weights ./ sum(weights)

    # output
    output = ABCRejectionOutput(input.n_params,
                                n_accepted,
                                n_tries,
                                input.threshold,
                                accepted_parameters,
                                accepted_distances,
                                StatsBase.Weights(weights, 1.0),
                                )

    #write(out_stream, output)

    return output

end

#
# Emulated version
#
function ABCrejection(
	input::EmulatedABCRejectionInput,
	reference_data;
	out_stream::IO = STDOUT,
    write_progress = true,
    progress_every = 1000)

	checkABCInput(input)

	# initialise
    n_accepted = 0
    n_tries = 0
    batch_no = 1
    accepted_parameters = zeros(input.n_particles, input.n_params)
    accepted_distances = zeros(input.n_particles)
    weights = ones(input.n_particles)

    # emulate
    while n_accepted < input.n_particles

        if batch_no > input.max_iter
            warn("Emulation reached maximum iterations before finding $(input.n_particles) particles - will return $n_accepted")
            accepted_parameters = accepted_parameters[1:n_accepted,:]
            accepted_distances = accepted_distances[1:n_accepted]
            weights = weights[1:n_accepted]

            break
        end

        parameter_batch, weight_batch = generate_parameters(input.priors, input.batch_size)

        distances = input.distance_prediction_function(parameter_batch)
        n_tries += input.batch_size

        #
        # Check which parameter indices were accepted
        #
        accepted_batch_idxs = find(distances .<= input.threshold)
        #println("accepted_batch_idxs = $(accepted_batch_idxs)")
        n_accepted_batch = length(accepted_batch_idxs)

        #
        # If some parameters were accepted, store their values, (predicted) distances and weights
        #
        if n_accepted_batch > 0
            # Check that we won't accept too many parameters
            if n_accepted + n_accepted_batch > input.n_particles
                accepted_batch_idxs = accepted_batch_idxs[1:input.n_particles - n_accepted]
                n_accepted_batch = length(accepted_batch_idxs)
            end

            accepted_parameters[n_accepted+1:n_accepted + n_accepted_batch,:] = parameter_batch[accepted_batch_idxs,:]
            accepted_distances[n_accepted+1:n_accepted + n_accepted_batch] = distances[accepted_batch_idxs]
            weights[n_accepted+1:n_accepted + n_accepted_batch] = weight_batch[accepted_batch_idxs]
            n_accepted += n_accepted_batch

        end

        if write_progress && (batch_no % progress_every == 0)
            write(out_stream, string(DateTime(now())),
                              " Accepted ",
                              string(n_accepted),
                              "/",
                              string(n_tries),
                              " particles (",
                              string(batch_no),
                              " batches of size ",
                              string(input.batch_size),
                              ").\n"
                              )
            flush(out_stream)
        end

        batch_no += 1
    end

    weights = weights ./ sum(weights)

    # output
    output = ABCRejectionOutput(input.n_params,
                                n_accepted,
                                n_tries,
                                input.threshold,
                                accepted_parameters,
                                accepted_distances,
                                StatsBase.Weights(weights, 1.0),
                                )

    #write(out_stream, output)

    return output

end
