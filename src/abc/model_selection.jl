"""
	model_selection(input::ModelSelectionInput,
		reference_data::AbstractArray{Float64,2})

# Arguments
- `input::ModelSelectionInput`: A ['ModelSelectionInput']@(ref) object that contains the settings for the model selection algorithm. 
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
"""
function model_selection(
	input::ModelSelectionInput,
	reference_data::AbstractArray{Float64,2})

	# For logging
	log_prefix = "GpABC model selection simulation "
	if isa(input, EmulatedModelSelectionInput)
		log_prefix = "GpABC model selection emulation "
	end

	info(string(DateTime(now())),
		" Population 1 - ABC Rejection ϵ = $(input.threshold_schedule[1])",
		prefix=log_prefix)

	tracker = initialise_modelselection(input, reference_data)

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
		iterate_modelselection!(tracker, input.threshold_schedule[i], reference_data)

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
function initialise_modelselection(input::SimulatedModelSelectionInput, reference_data::AbstractArray{Float64,2})

	#
	# Check input sizes
	#
	if span(input.model_prior) != input.M
		throw(ArgumentError("There are $(input.M) models but the span of the model prior support is $(span(input.model_prior))"))
	end

	if length(input.simulator_functions) != input.M
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

    # Summary statistic initialisation
    built_summary_statistic = build_summary_statistic(input.summary_statistic)
    summarised_reference_data = built_summary_statistic(reference_data)

    # Initialise variables to hold ABC results
    parameters = zeros(0)
    distance = 0.0
    weight_value = 0.0

	#
	# Compute first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter

		m = rand(input.model_prior)

		try
            parameters, distance, weight_value = check_particle(input.parameter_priors[m],
            													input.simulator_functions[m],
                                                                built_summary_statistic,
                                                                input.distance_function,
                                                                summarised_reference_data)
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
		if distance < input.threshold_schedule[1]
			total_n_accepted += 1
			update_candidatemodeltracker!(cm_trackers[m], parameters, distance, weight_value)
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
		built_summary_statistic,
		summarised_reference_data,
		input.distance_function,
		input.simulator_functions[m],
		1) for m in 1:input.M]

	return SimulatedModelSelectionTracker(input.M,
		input.n_particles,
		[input.threshold_schedule[1]],
		input.model_prior,
		smc_trackers,
		built_summary_statistic,
		input.distance_function,
		input.max_iter)
end

# Initialises the model selection run and runs the first (rejection) population
function initialise_modelselection(input::EmulatedModelSelectionInput, reference_data::AbstractArray{Float64,2})

	#
	# Check input sizes
	#
	if span(input.model_prior) != input.M
		throw(ArgumentError("There are $(input.M) models but the span of the model prior support is $(span(input.model_prior))"))
	end

	if length(input.emulator_trainers) != input.M
		throw(ArgumentError("There are $(input.M) models but $(length(input.emulation_settings_arr)) emulation settings"))
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
	prior_samplers = [f(n_design_points) = generate_parameters(input.parameter_priors[m], n_design_points)[1] for m in 1:input.M]
	emulators = [input.emulator_trainers[m](prior_samplers[m]) for m in 1:input.M]

    # Initialise variables to hold ABC results
    total_n_accepted = 0
	n_iterations = 1
    parameter_batch = zeros(0,0)
    distance_batch = zeros(0)
    vars_batch = zeros(0)
    weight_values_batch = zeros(0)

	#
	# Emulate first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter
		sampled_models = rand(input.model_prior, min(input.max_batch_size, input.n_particles-total_n_accepted))
		tries_this_it = [size(sampled_models[sampled_models.==m],1) for m=1:input.M]

		for m=1:input.M
			if tries_this_it[m] > 0

				parameter_batch, weight_values_batch, distance_batch, vars_batch = check_particle_batch(input.parameter_priors[m],
																										 tries_this_it[m],
																										 emulators[m])

				cm_trackers[m].n_tries += tries_this_it[m]

				accepted_batch_idxs = find_accepted_particle_idxs(distance_batch,
																  vars_batch,
																  input.threshold_schedule[1])


				# If at least one particle is accepted
				if length(accepted_batch_idxs) > 0
					total_n_accepted += length(accepted_batch_idxs)
					update_candidatemodeltracker!(cm_trackers[m],
												  accepted_batch_idxs,
												  parameter_batch,
												  distance_batch,
												  weight_values_batch)

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

	smc_trackers = [EmulatedABCSMCTracker(
		length(input.parameter_priors[m]),
		[cm_trackers[m].n_accepted],
		[cm_trackers[m].n_tries],
		[input.threshold_schedule[1]],
		[cm_trackers[m].population],
		[cm_trackers[m].distances],
		[StatsBase.Weights(cm_trackers[m].weight_values ./ sum(cm_trackers[m].weight_values), 1.0)],
		input.parameter_priors[m],
		input.emulator_trainers[m],
		1,
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
	threshold::AbstractFloat,
	reference_data::AbstractArray{Float64,2})

	# Generate kernels for alive models only
	kernels = Dict{Integer,Matrix{ContinuousUnivariateDistribution}}()
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

		if tracker.smc_trackers[m].n_accepted[end] == 0
			continue
		end

		# Do ABC SMC for a single particle
		try
            parameters, distance, weight_value = check_particle(tracker.smc_trackers[m],
            													kernels[m])
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

		if distance < threshold
			total_n_accepted += 1
			update_candidatemodeltracker!(cm_trackers[m], parameters, distance, weight_value)
		end

		n_iterations += 1

	end

	if n_iterations == tracker.max_iter+1
		warn("Simulated model selection reached maximum number of iterations ($(tracker.max_iter)) on the an SMC run - consider trying more iterations.")
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
	tracker::EmulatedModelSelectionTracker,
	threshold::AbstractFloat,
	reference_data::AbstractArray{Float64,2})

	# Initialise
	cm_trackers = [CandidateModelTracker(length(smc_tracker.priors)) for smc_tracker in tracker.smc_trackers]

	# Generate kernels for alive models only
	kernels = Dict{Integer,Matrix{ContinuousUnivariateDistribution}}()
	for m in 1:tracker.M
		if tracker.smc_trackers[m].n_accepted[end] > 0
			kernels[m] = generate_kernels(tracker.smc_trackers[m].population[end],
										  tracker.smc_trackers[m].priors)
		end
	end

	# Retrain emulators for alive models only
	#retrained_emulators = Array{GPModel,1}(tracker.M)
	retrained_emulators = [smc_tracker.emulators[end] for smc_tracker in tracker.smc_trackers]
	# for m in 1:tracker.M
	# 	if tracker.smc_trackers[m].n_accepted[end] > 0
	# 		retrained_emulators[m] = retrain_emulator(tracker.smc_trackers[m])
	# 	else
	# 		retrained_emulators[m] = GPModel()
	# 	end
	# end

    # Initialise
    total_n_accepted = 0
	n_iterations = 1
    parameter_batch = zeros(0,0)
    weight_values_batch = zeros(0)
    distance_batch = zeros(0)
    vars_batch = zeros(0)

    # emulate
	while total_n_accepted < tracker.n_particles && n_iterations <= tracker.max_iter

		sampled_models = rand(tracker.model_prior, min(tracker.max_batch_size, tracker.n_particles-total_n_accepted))
		tries_this_it = [size(sampled_models[sampled_models.==m],1) for m=1:tracker.M]

		for m = 1:tracker.M
			# Skip a model that wasn't sampled in this iteration or had no accepted particles in the previous population
			if tries_this_it[m] == 0 || tracker.smc_trackers[m].n_accepted[end] == 0
				continue
			end

			parameter_batch, weight_values_batch, distance_batch, vars_batch = check_particle_batch(tracker.smc_trackers[m],
																								   kernels[m],
																								   retrained_emulators[m],
																								   tries_this_it[m])
			cm_trackers[m].n_tries += tries_this_it[m]

			accepted_batch_idxs = find_accepted_particle_idxs(distance_batch,
															vars_batch,
															threshold)


			# If at least one particle is accepted
			if length(accepted_batch_idxs) > 0
				total_n_accepted += length(accepted_batch_idxs)
				update_candidatemodeltracker!(cm_trackers[m],
												  accepted_batch_idxs,
												  parameter_batch,
												  distance_batch,
												  weight_values_batch)

			elseif length(accepted_batch_idxs) > tries_this_it[m]
				error("$(length(accepted_batch_idxs)) particles were accepted when model $m was only sampled $(tries_this_it[m]) times!")
			end																				   
		end

		n_iterations += 1
	end

	if n_iterations == tracker.max_iter+1
		warn("Emulated model selection reached maximum number of iterations ($(tracker.max_iter)) on the an SMC run - consider trying more iterations.")
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
	parameters::AbstractArray{Float64,1},
	distance::AbstractFloat,
	weight_value::AbstractFloat)

	cm_tracker.n_accepted += 1
	cm_tracker.population = vcat(cm_tracker.population, parameters')
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
	cm_tracker.distances = vcat(cm_tracker.distances, distance_batch[accepted_batch_idxs])
	cm_tracker.weight_values = vcat(cm_tracker.weight_values, weight_values_batch[accepted_batch_idxs])
end

# not exported
function update_modelselection_tracker!(
	tracker::SimulatedModelSelectionTracker,
	cm_trackers::AbstractArray{CandidateModelTracker,1},
	threshold::AbstractFloat)

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
	cm_trackers::AbstractArray{CandidateModelTracker,1},
	retrained_emulators::AbstractArray{GPModel,1},
	threshold::AbstractFloat)

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