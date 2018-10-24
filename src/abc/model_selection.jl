"""
	model_selection(input::ModelSelectionInput,
		reference_data::AbstractArray{Float64,2})

# Arguments
- `input::ModelSelectionInput`: A ['ModelSelectionInput']@(ref) object that contains the settings for the model selection algorithm.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
"""
function model_selection(input::ModelSelectionInput)

	# For logging
	log_prefix = "GpABC model selection simulation "
	if isa(input, EmulatedModelSelectionInput)
		log_prefix = "GpABC model selection emulation "
	end

	info(string(DateTime(now())),
		" Population 1 - ABC Rejection ϵ = $(input.threshold_schedule[1])",
		prefix=log_prefix)

	tracker = initialise_modelselection(input)

	if all_models_dead(tracker)
		warn("No particles were accepted in population 1 with threshold $(input.threshold_schedule[1])- terminating model selection algorithm")
		return build_modelselection_output(tracker, false)
	end

	if all_but_one_models_dead(tracker)
		warn("All but one model is dead after population 1 - terminating model selection algorithm")
		return build_modelselection_output(tracker, true)
	end

	for i in 2:length(input.threshold_schedule)
		info(string(DateTime(now())),
			" Population $i - ABC SMC ϵ = $(input.threshold_schedule[i])",
			prefix=log_prefix)
		iterate_modelselection!(tracker, input.threshold_schedule[i])

		# Avoid infinite loop if no particles are accepted
		if all_models_dead(tracker)
			warn("No particles were accepted in population $i with threshold $(input.threshold_schedule[i])- terminating model selection algorithm")
			return build_modelselection_output(tracker, false)
		end

		if all_but_one_models_dead(tracker)
			warn("All but one model is dead after population $i - terminating model selection algorithm")
			return build_modelselection_output(tracker, true)
		end
	end

	return build_modelselection_output(tracker, true)
end

# Initialises the model selection run and runs the first (rejection) population
function initialise_modelselection(input::SimulatedModelSelectionInput)

	#
	# Check input sizes
	#
	if span(input.model_prior) != input.M
		throw(ArgumentError("There are $(input.M) models but the span of the model prior support is $(span(input.model_prior))"))
	end

	if length(input.distance_simulation_input) != input.M
		throw(ArgumentError("There are $(input.M) models but $(length(input.simulator_functions)) simulator functions"))
	end

	if length(input.parameter_priors) != input.M
		throw(ArgumentError("There are $(input.M) models but $(length(input.parameter_priors)) sets of parameter priors"))
	end

	total_n_accepted = 0
	n_iterations = 1

	#
	# Initialise arrays that will track rejection ABC run for each model - these
	# will be used to create SimulatedCandidateModelTrackers after the rejection ABC run
	#
	cm_trackers = [CandidateModelTracker(length(input.parameter_priors[m]))
								for m in 1:input.M]

    # Initialise variables to hold ABC results
    parameters = zeros(0)
    distance = 0.0
    weight_value = 0.0

	#
	# Compute first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter

		m = rand(input.model_prior)
		parameters, weight_value = generate_parameters(input.parameter_priors[m])
		try
			distance = simulate_distance(parameters, input.distance_simulation_input[m])
        catch e
            if isa(e, DimensionMismatch)
                # This prevents the whole code from failing if there is a problem
                # solving the differential equation(s). The exception is thrown by the
                # distance function
                warn("The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of DifferentialEquations::solve - please check for related warnings. Continuing to the next iteration.")
                n_iterations += 1
                continue
            else
                throw(e)
            end
        end

        cm_trackers[m].n_tries += 1

		# If particle accepted
		if distance[1] < input.threshold_schedule[1]
			total_n_accepted += 1
			update_candidatemodeltracker!(cm_trackers[m], parameters, distance[1], weight_value)
		end

		n_iterations += 1
	end

	if n_iterations == input.max_iter+1
		warn("Simulated model selection reached maximum number of iterations ($(input.max_iter)) on the first population - consider trying more iterations.")
	end

	info(string(DateTime(now())),
		" Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total.\n",
		"Number of accepted parameters by model: ",
		join([string("Model ", m, ": ",  cm_trackers[m].n_accepted) for m in 1:input.M], "\t"),
		prefix="GpABC model selection simulation ",
	)

	smc_trackers = [SimulatedABCSMCTracker(
		length(input.parameter_priors[m]),
		[cm_trackers[m].n_accepted],
		[cm_trackers[m].n_tries],
		[input.threshold_schedule[1]],
		[cm_trackers[m].population],
		[cm_trackers[m].distances],
		[StatsBase.Weights(cm_trackers[m].weight_values ./ sum(cm_trackers[m].weight_values), 1.0)], 	# Normalise weights for each model
		input.parameter_priors[m],
		input.distance_simulation_input[m],
		1) for m in 1:input.M]

	return SimulatedModelSelectionTracker(input.M,
		input.n_particles,
		[input.threshold_schedule[1]],
		input.model_prior,
		smc_trackers,
		input.max_iter)
end

# Initialises the model selection run and runs the first (rejection) population
function initialise_modelselection(input::EmulatedModelSelectionInput{AF, CUD, ER, EPS, ET}) where {
	AF<:AbstractFloat, CUD<:ContinuousUnivariateDistribution,
	ER<:AbstractEmulatorRetraining, EPS<:AbstractEmulatedParticleSelection, ET<:AbstractEmulatorTraining}
	#
	# Check input sizes
	#
	if span(input.model_prior) != input.M
		throw(ArgumentError("There are $(input.M) models but the span of the model prior support is $(span(input.model_prior))"))
	end

	if length(input.parameter_priors) != input.M
		throw(ArgumentError("There are $(input.M) models but $(length(input.parameter_priors)) sets of parameter priors"))
	end

	#
	# Initialise arrays that will track rejection ABC run for each model - these
	# will be used to create CandidateModelTrackers after the rejection ABC run
	#
	cm_trackers = [CandidateModelTracker(length(input.parameter_priors[m]))
								for m in 1:input.M]

	#
	# Train the emulators
	#
	emulators = []
	for m in 1:input.M
		emulator = abc_train_emulator(input.parameter_priors[m], input.emulator_training_input[m])
		push!(emulators, emulator)
	end

    # Initialise variables to hold ABC results
    total_n_accepted = 0
	n_iterations = 1

	#
	# Emulate first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter
		sampled_models = rand(input.model_prior, min(input.max_batch_size, input.n_particles-total_n_accepted))
		tries_this_it = [size(sampled_models[sampled_models.==m],1) for m=1:input.M]

		for m=1:input.M
			if tries_this_it[m] > 0

				parameters, weights = generate_parameters(input.parameter_priors[m], tries_this_it[m])
				distances, accepted_batch_idxs = abc_select_emulated_particles(emulators[m], parameters,
				 	input.threshold_schedule[1], input.emulated_particle_selection)
				cm_trackers[m].n_tries += tries_this_it[m]

				# If at least one particle is accepted
				if length(accepted_batch_idxs) > 0
					total_n_accepted += length(accepted_batch_idxs)
					update_candidatemodeltracker!(cm_trackers[m],
												  accepted_batch_idxs,
												  parameters,
												  distances,
												  weights)

				elseif length(accepted_batch_idxs) > tries_this_it[m]
					error("$(length(accepted_batch_idxs))) particles were accepted when model $m was only sampled $(tries_this_it[m]) times!")
				end
			end
		end

		n_iterations += 1
	end

	if n_iterations == input.max_iter+1
		warn("Emulated model selection reached maximum number of iterations ($(input.max_iter)) without accepting $(input.n_particles) particles on the first population - consider trying more iterations.")
	end

	info(string(DateTime(now())),
		" Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total.\n",
		"Number of accepted parameters by model: ",
		join([string("Model ", m, ": ",  cm_trackers[m].n_accepted) for m in 1:input.M], "\t"),
		prefix="GpABC model selection emulation ",
	)

	smc_trackers = [EmulatedABCSMCTracker{CUD, typeof(emulators[m]), ER, EPS}(
		length(input.parameter_priors[m]),
		[cm_trackers[m].n_accepted],
		[cm_trackers[m].n_tries],
		[input.threshold_schedule[1]],
		[cm_trackers[m].population],
		[cm_trackers[m].distances],
		[StatsBase.Weights(cm_trackers[m].weight_values ./ sum(cm_trackers[m].weight_values), 1.0)],
		input.parameter_priors[m],
		input.emulator_training_input[m],
		input.emulator_retraining,
		input.emulated_particle_selection,
		input.max_batch_size,
		input.max_iter,
		[emulators[m]]) for m = 1:input.M]

	return EmulatedModelSelectionTracker(
				input.M,
				input.n_particles,
				[input.threshold_schedule[1]],
				input.model_prior,
				smc_trackers,
				input.max_batch_size,
				input.max_iter)
end

# Perform a subsequent model selection iteration (based on ABC-SMC)
function iterate_modelselection!(tracker::SimulatedModelSelectionTracker,
	threshold::AbstractFloat)

	# Generate kernels for alive models only
	kernels = Dict{Int,Matrix{ContinuousUnivariateDistribution}}()
	for m in 1:tracker.M
		if tracker.smc_trackers[m].n_accepted[end] > 0
			kernels[m] = generate_kernels(tracker.smc_trackers[m].population[end],
										  tracker.smc_trackers[m].priors)
		end
	end

	# Initialise
	cm_trackers = [CandidateModelTracker(tracker.smc_trackers[m].n_params) for m in 1:tracker.M]
	total_n_accepted = 0
	n_iterations = 1
	parameters = zeros(0)
	distance = 0.0
	weight_value = 0.0

	while total_n_accepted < tracker.n_particles && n_iterations <= tracker.max_iter
		# Sample model from prior
		m = rand(tracker.model_prior)
		smc_tracker = tracker.smc_trackers[m]

		if smc_tracker.n_accepted[end] == 0
			continue
		end

		# Do ABC SMC for a single particle
		parameters, weight_value = generate_parameters(smc_tracker.priors, smc_tracker.weights[end], kernels[m])
		try
            distance = simulate_distance(parameters, smc_tracker.distance_simulation_input)
        catch e
            if isa(e, DimensionMismatch)
                # This prevents the whole code from failing if there is a problem
                # solving the differential equation(s). The exception is thrown by the
                # distance function
                warn("The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of DifferentialEquations::solve - please check for related warnings. Continuing to the next iteration.")
                n_iterations += 1
                continue
            else
                throw(e)
            end
        end

        cm_trackers[m].n_tries += 1

		if distance[1] < threshold
			total_n_accepted += 1
			update_candidatemodeltracker!(cm_trackers[m], parameters, distance[1], weight_value)
		end

		n_iterations += 1

	end

	if n_iterations == tracker.max_iter+1
		warn("Simulated model selection reached maximum number of iterations ($(tracker.max_iter)) on an SMC run - consider trying more iterations.")
	end

	info(string(DateTime(now())),
		" Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total.\n",
		"Number of accepted parameters by model: ",
		join([string("Model ", m, ": ",  cm_trackers[m].n_accepted) for m in 1:tracker.M], "\t"),
		prefix="GpABC model selection simulation "
	)

	update_modelselection_tracker!(tracker, cm_trackers, threshold)
end

function iterate_modelselection!(
	tracker::EmulatedModelSelectionTracker{CUD, ET, ER, EPS},
	threshold::AbstractFloat) where {CUD<:ContinuousUnivariateDistribution, ET,
	        ER<:AbstractEmulatorRetraining, EPS<:AbstractEmulatedParticleSelection}

	# Initialise
	cm_trackers = [CandidateModelTracker(length(smc_tracker.priors)) for smc_tracker in tracker.smc_trackers]

	# Generate kernels for alive models only
	kernels = Dict{Int,Matrix{ContinuousUnivariateDistribution}}()
	for m in 1:tracker.M
		if tracker.smc_trackers[m].n_accepted[end] > 0
			kernels[m] = generate_kernels(tracker.smc_trackers[m].population[end],
										  tracker.smc_trackers[m].priors)
		end
	end

	# Retrain emulators for alive models only
	retrained_emulators = Array{ET,1}(undef,tracker.M)
	for m in 1:tracker.M
		smc_tracker = tracker.smc_trackers[m]
		if smc_tracker.n_accepted[end] > 0
			particle_sampling_function(batch_size) = generate_parameters_no_weights(batch_size,
		        smc_tracker.population[end],
		        smc_tracker.weights[end],
		        kernels[m])
			retrained_emulators[m] = abc_retrain_emulator(smc_tracker.emulators[end],
				particle_sampling_function,
				threshold,
				smc_tracker.emulator_training_input,
				smc_tracker.emulator_retraining)
		else
			retrained_emulators[m] = smc_tracker.emulators[end]
		end
	end

    # Initialise
    total_n_accepted = 0
	n_iterations = 1

    # emulate
	while total_n_accepted < tracker.n_particles && n_iterations <= tracker.max_iter

		sampled_models = rand(tracker.model_prior, min(tracker.max_batch_size, tracker.n_particles-total_n_accepted))
		tries_this_it = [size(sampled_models[sampled_models.==m],1) for m=1:tracker.M]

		for m = 1:tracker.M
			# Skip a model that wasn't sampled in this iteration or had no accepted particles in the previous population
			if tries_this_it[m] == 0 || tracker.smc_trackers[m].n_accepted[end] == 0
				continue
			end
			smc_tracker = tracker.smc_trackers[m]
			parameters, weights = generate_parameters(tries_this_it[m],
	                                                 smc_tracker.priors,
	                                                 smc_tracker.weights[end],
	                                                 kernels[m])
			cm_trackers[m].n_tries += tries_this_it[m]

			distances, accepted_indices = abc_select_emulated_particles(retrained_emulators[m], parameters, threshold, smc_tracker.selection)


			# If at least one particle is accepted
			if length(accepted_indices) > 0
				total_n_accepted += length(accepted_indices)
				update_candidatemodeltracker!(cm_trackers[m],
												  accepted_indices,
												  parameters,
												  distances,
												  weights)

			elseif length(accepted_indices) > tries_this_it[m]
				error("$(length(accepted_indices)) particles were accepted when model $m was only sampled $(tries_this_it[m]) times!")
			end
		end

		n_iterations += 1
	end

	if n_iterations == tracker.max_iter+1
		warn("Emulated model selection reached maximum number of iterations ($(tracker.max_iter)) on an SMC run - consider trying more iterations.")
	end

	info(string(DateTime(now())),
		" Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total.\n",
		"Number of accepted parameters by model: ",
		join([string("Model ", m, ": ",  cm_trackers[m].n_accepted) for m in 1:tracker.M], "\t"),
		prefix="GpABC model selection emulation ",
	)

	update_modelselection_tracker!(tracker, cm_trackers, retrained_emulators, threshold)
end

# not exported
function build_modelselection_output(tracker::ModelSelectionTracker, successful_completion::Bool)
	return ModelSelectionOutput(
		tracker.M,
		[[tracker.smc_trackers[m].n_accepted[i] for m = 1:tracker.M] for i in 1:length(tracker.threshold_schedule)],
		tracker.threshold_schedule,
		[buildAbcSmcOutput(tracker.smc_trackers[m]) for m in 1:tracker.M],
		successful_completion
		)
end

# not exported
function all_models_dead(tracker::ModelSelectionTracker)
	return sum([tracker.smc_trackers[m].n_accepted[end] for m = 1:tracker.M]) == 0
end

# not exported
function all_but_one_models_dead(tracker::ModelSelectionTracker)
	return sum([tracker.smc_trackers[m].n_accepted[end] == 0 for m = 1:tracker.M]) == tracker.M-1
end

# not exported
function update_candidatemodeltracker!(
	cm_tracker::CandidateModelTracker,
	parameters::AbstractArray{Float64,2},
	distance::AbstractFloat,
	weight_value::AbstractFloat)

	cm_tracker.n_accepted += 1
	cm_tracker.population = vcat(cm_tracker.population, parameters)
	push!(cm_tracker.distances, distance)
	push!(cm_tracker.weight_values, weight_value)
end

function update_candidatemodeltracker!(
	cm_tracker::CandidateModelTracker,
	accepted_batch_idxs::AbstractArray{Int64,1},
	parameter_batch::AbstractArray{Float64,2},
	distance_batch::AbstractArray{Float64,1},
	weight_values_batch::AbstractArray{Float64,1})

	cm_tracker.n_accepted += length(accepted_batch_idxs)
	cm_tracker.population = vcat(cm_tracker.population, parameter_batch[accepted_batch_idxs,:])
	cm_tracker.distances = vcat(cm_tracker.distances, distance_batch)
	cm_tracker.weight_values = vcat(cm_tracker.weight_values, weight_values_batch[accepted_batch_idxs])
end

# not exported
function update_modelselection_tracker!(
	tracker::SimulatedModelSelectionTracker,
	cm_trackers::AbstractArray{CandidateModelTracker{AF},1},
	threshold::AF) where {AF<:AbstractFloat}
	push!(tracker.threshold_schedule, threshold)
	for m in 1:tracker.M
		push!(tracker.smc_trackers[m].n_accepted, cm_trackers[m].n_accepted)
		push!(tracker.smc_trackers[m].n_tries, cm_trackers[m].n_tries)
		push!(tracker.smc_trackers[m].threshold_schedule, threshold)
		push!(tracker.smc_trackers[m].population, cm_trackers[m].population)
		push!(tracker.smc_trackers[m].distances, cm_trackers[m].distances)
		push!(tracker.smc_trackers[m].weights, normalise(cm_trackers[m].weight_values))
	end
end

# not exported
function update_modelselection_tracker!(
	tracker::EmulatedModelSelectionTracker,
	cm_trackers::AbstractArray{CandidateModelTracker{AF},1},
	retrained_emulators::AbstractArray{GPModel,1},
	threshold::AF) where {AF<:AbstractFloat}

	push!(tracker.threshold_schedule, threshold)
	for m in 1:tracker.M
		push!(tracker.smc_trackers[m].n_accepted, cm_trackers[m].n_accepted)
		push!(tracker.smc_trackers[m].n_tries, cm_trackers[m].n_tries)
		push!(tracker.smc_trackers[m].threshold_schedule, threshold)
		push!(tracker.smc_trackers[m].population, cm_trackers[m].population)
		push!(tracker.smc_trackers[m].distances, cm_trackers[m].distances)
		push!(tracker.smc_trackers[m].weights, normalise(cm_trackers[m].weight_values))
		push!(tracker.smc_trackers[m].emulators, retrained_emulators[m])
	end
end
