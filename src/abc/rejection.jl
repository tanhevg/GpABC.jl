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

"""
    ABCrejection

Run a simulationed-based rejection-ABC computation. Parameter posteriors are obtained by simulating the model
for a parameter vector, computing the summary statistic of the output then computing the distance to the
summary statistic of the reference data. If this distance is sufficiently small the parameter vector is
included in the posterior.

# Arguments
- `input::SimulatedABCRejectionInput`: A ['SimulatedABCRejectionInput'](@ref) object that defines the settings for the simulated rejection-ABC run.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `write_progress::Bool`: Optional argument controlling whether progress is written to `out_stream`.
- `progress_every::Int`: Progress will be written to `out_stream` every `progress_every` simulations (optional, ignored if `write_progress` is `False`).

# Returns
A ['SimulatedABCRejectionOutput'](@ref) object.
"""
function ABCrejection(input::SimulatedABCRejectionInput;
    write_progress::Bool = true,
    progress_every::Int = 1000)

	checkABCInput(input)
    if write_progress
        info(string(DateTime(now())), " ϵ = $(input.threshold)."; prefix="GpABC rejection simulation ")
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
                warn("The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of DifferentialEquations::solve - please check for related warnings. Continuing to the next iteration.")
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
            info(string(DateTime(now())), " Accepted $(n_accepted)/$(n_tries) particles.", prefix="GpABC rejection simulation ")
        end
    end

    if n_accepted < input.n_particles
        warn("Simulation reached maximum iterations $(input.max_iter) before finding $(input.n_particles) particles - will return $n_accepted")
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

"""
    ABCrejection
Run a emulation-based rejection-ABC computation. Parameter posteriors are obtained using a regression model
(the emulator), that has learnt a mapping from parameter vectors to the distance between the
model output and observed data in summary statistic space. If this distance is sufficiently small the parameter vector is
included in the posterior.
# Arguments
- `input::EmulatedABCRejectionInput`: An ['EmulatedABCRejectionInput'](@ref) object that defines the settings for the emulated rejection-ABC run.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
- `write_progress::Bool`: Optional argument controlling whether progress is logged.
- `progress_every::Int`: Progress will be logged every `progress_every` simulations (optional, ignored if `write_progress` is `False`).
# Returns
An ['EmulatedABCRejectionOutput'](@ref) object.
"""
function ABCrejection(input::EmulatedABCRejectionInput;
    write_progress = true,
    progress_every = 1000)

    checkABCInput(input)

    if write_progress
        info(string(DateTime(now())), " ϵ = $(input.threshold).", prefix="GpABC rejection emulation ")
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
            info(string(DateTime(now())),
                " accepted $(n_accepted)/$(n_tries) particles ($(batch_no) batches of size $(input.batch_size)).",
                prefix="GpABC rejection emulation ")
        end

        batch_no += 1
    end

    if n_accepted < input.n_particles
        warn("Emulation reached maximum $(input.max_iter) iterations before finding $(input.n_particles) particles - will return $n_accepted")
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
