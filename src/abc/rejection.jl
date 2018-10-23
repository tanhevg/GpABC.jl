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

    return parameters, weight
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
    weights = prod(pdf.(priors, parameters), 2)
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
function ABCrejection(
    input::SimulatedABCRejectionInput,
	reference_data::AbstractArray{Float64,2};
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
    weight_values = ones(input.n_particles)

    parameters = zeros(input.n_params)
    distance = 0.0
    weight_value = 0.0

    # Summary statistic initialisation
    built_summary_statistic = build_summary_statistic(input.summary_statistic)
    summarised_reference_data = built_summary_statistic(reference_data)

    # simulate
    while n_accepted < input.n_particles && n_tries < input.max_iter

        try
            parameters, distance, weight_value = check_particle(input.priors,
                                                                input.simulator_function,
                                                                built_summary_statistic,
                                                                input.distance_function,
                                                                summarised_reference_data)
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

        if distance <= input.threshold
            n_accepted += 1
            accepted_parameters[n_accepted,:] = parameters
            accepted_distances[n_accepted] = distance
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
function ABCrejection(
    input::EmulatedABCRejectionInput,
    reference_data::AbstractArray{Float64,2},
    batch_size::Integer;
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
    weight_values = ones(input.n_particles)

    # todo: consolidate sample_from_priors with generate_parameters
    # prior_sampling_function(n_design_points) = generate_parameters(input.priors, n_design_points)[1]
    #
    # emulator = input.train_emulator_function(prior_sampling_function)

    # todo: consolidate sample_from_priors with generate_parameters
    prior_sampling_function(n_design_points) = generate_parameters(input.priors, n_design_points)[1]
    emulator = input.train_emulator_function(prior_sampling_function)

    # emulate
    while n_accepted < input.n_particles && batch_no <= input.max_iter

        parameter_batch, weight_values_batch, distances, vars = check_particle_batch(input.priors,
                                                                                     batch_size,
                                                                                     emulator)
        n_tries += batch_size

        #
        # Check which parameter indices were accepted
        #
        accepted_batch_idxs = find_accepted_particle_idxs(distances, vars, input.threshold)
        n_accepted_batch = length(accepted_batch_idxs)

        #
        # If some parameters were accepted, store their values, (predicted) distances and weights
        #
        if n_accepted_batch > 0
            # Check that we won't accept too many parameters - throw away the extra parameters if so
            if n_accepted + n_accepted_batch > input.n_particles
                accepted_batch_idxs = accepted_batch_idxs[1:input.n_particles - n_accepted]
                n_accepted_batch = length(accepted_batch_idxs)
            end

            accepted_parameters[n_accepted+1:n_accepted + n_accepted_batch,:] = parameter_batch[accepted_batch_idxs,:]
            accepted_distances[n_accepted+1:n_accepted + n_accepted_batch] = distances[accepted_batch_idxs]
            weight_values[n_accepted+1:n_accepted + n_accepted_batch] = weight_values_batch[accepted_batch_idxs]
            n_accepted += n_accepted_batch

        end

        if write_progress
            info(string(DateTime(now())),
                " accepted $(n_accepted)/$(n_tries) particles ($(batch_no) batches of size $(batch_size)).",
                prefix="GpABC rejection emulation ")
        end

        batch_no += 1
    end

    if n_accepted < input.n_particles
        warn("Emulation reached maximum $(input.max_iter) iterations before finding $(input.n_particles) particles - will return $n_accepted")
        accepted_parameters = accepted_parameters[1:n_accepted, :]
        accepted_distances = accepted_distances[1:n_accepted]
        weight_values = weight_values[1:n_accepted]
    end

    # output
    output = EmulatedABCRejectionOutput(input.n_params,
                                n_accepted,
                                n_tries,
                                input.threshold,
                                accepted_parameters,
                                accepted_distances,
                                StatsBase.Weights(weight_values ./ sum(weight_values)),
                                emulator
                                )

    return output
end

# not exported
function check_particle(
    priors::AbstractArray{CUD,1},
    simulator_function::Function,
    built_summary_statistic::Function,
    distance_function::Function,
    summarised_reference_data::AbstractArray{Float64,1}) where {
    CUD <: ContinuousUnivariateDistribution
    }

    parameters, weight_value = generate_parameters(priors)
    simulated_data = simulator_function(parameters)
    summarised_simulated_data = built_summary_statistic(simulated_data)
    distance = distance_function(summarised_reference_data, summarised_simulated_data)
    
    return parameters, distance, weight_value
end

# not exported
function check_particle_batch(
    priors::AbstractArray{CUD,1},
    batch_size::Integer,
    emulator::GPModel) where {
    CUD <: ContinuousUnivariateDistribution
    }

    parameter_batch, weight_values_batch = generate_parameters(priors, batch_size)
    distance_batch, var_batch = gp_regression(parameter_batch, emulator)

    return parameter_batch, weight_values_batch, distance_batch, var_batch
end

# not exported
function find_accepted_particle_idxs(
    distances::AbstractArray{Float64,1},
    vars::AbstractArray{Float64,1},
    threshold::AbstractFloat)

    return find((distances .<= threshold) .& (sqrt.(vars) .<= threshold))
end