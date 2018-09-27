#
# SIMULATION
#

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
	rejection_trackers = [ModelSelectionRejectionTracker(length(input.parameter_priors[m]))
								for m in 1:input.M]

	#
	# Compute first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter
		m = rand(input.model_prior)
		rejection_trackers[m].n_tries += 1

		out = ABCrejection(
				SimulatedABCRejectionInput(length(input.parameter_priors[m]),
											1,
											input.threshold_schedule[1],
											input.parameter_priors[m],
											input.summary_statistic,
											input.distance_function,
											input.simulator_functions[m],
											1),
				reference_data,
				normalise_weights=false,
				for_model_selection=true)

		# If particle accepted
		if size(out.population,1) == 1
			total_n_accepted += 1
			rejection_trackers[m].n_accepted += 1
			rejection_trackers[m].population = vcat(rejection_trackers[m].population, out.population)
			push!(rejection_trackers[m].distances, out.distances[1])
			push!(rejection_trackers[m].weight_values, out.weights.values[1])
		elseif size(out.population, 1) > 1
			error("$(size(out.population,1)) particles were accepted!")
		end

		n_iterations += 1
	end

	println("Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total")

	if n_iterations == input.max_iter+1
		warn("Simulated model selection reached maximum number of iterations ($(input.max_iter)) on the first population - consider trying more iterations.")
	end

	# Normalise weights for each model
	for m=1:input.M
		rejection_trackers[m].weight_values = rejection_trackers[m].weight_values ./ sum(rejection_trackers[m].weight_values)
	end

	cmTrackers = [SimulatedCandidateModelTracker(
		length(input.parameter_priors[m]),
		[rejection_trackers[m].n_accepted],
		[rejection_trackers[m].n_tries],
		[rejection_trackers[m].population],
		[rejection_trackers[m].distances],
		[StatsBase.Weights(rejection_trackers[m].weight_values, 1.0)],
		input.parameter_priors[m],
		input.simulator_functions[m]
		)
			for m in 1:input.M]

		println("Number of accepted parameters: ", join([string("Model ", m, ": ",  cmTrackers[m].n_accepted[end]) for m in 1:input.M], "\t"))

	return SimulatedModelSelectionTracker(input.M,
		input.n_particles,
		[input.threshold_schedule[1]],
		input.model_prior,
		cmTrackers,
		build_summary_statistic(input.summary_statistic),
		input.distance_function,
		input.max_iter)
end

# Perform a subsequent model selection iteration (based on ABC-SMC)
function iterate_modelselection!(tracker::SimulatedModelSelectionTracker,
	threshold::AbstractFloat,
	reference_data::AbstractArray{Float64,2})

	push!(tracker.threshold_schedule, threshold)

	for mtracker in tracker.model_trackers
		push!(mtracker.n_accepted, 0)
		push!(mtracker.n_tries, 0)
		push!(mtracker.population, zeros(0, mtracker.n_params))
		push!(mtracker.distances, zeros(0))
		push!(mtracker.weights, StatsBase.Weights(zeros(0)))
	end

	total_n_accepted = 0
	n_iterations = 1

	while total_n_accepted < tracker.n_particles && n_iterations <= tracker.max_iter
		m = rand(tracker.model_prior)

		if tracker.model_trackers[m].n_accepted[end-1] == 0
			continue
		else
			tracker.model_trackers[m].n_tries[end] += 1
		end

		abcsmc_tracker = SimulatedABCSMCTracker(
				tracker.model_trackers[m].n_params,
				deepcopy(tracker.model_trackers[m].n_accepted[1:end-1]),
				[0],
				deepcopy(tracker.threshold_schedule[1:end-1]),
				deepcopy(tracker.model_trackers[m].population[1:end-1]),
				deepcopy(tracker.model_trackers[m].distances[1:end-1]),
				deepcopy(tracker.model_trackers[m].weights[1:end-1]),
				tracker.model_trackers[m].priors,
				tracker.summary_statistic,
				tracker.distance_function,
				tracker.model_trackers[m].simulator_function,
				1)

		particle_accepted = iterateABCSMC!(
			abcsmc_tracker,
			threshold,
			1,
			reference_data,
			normalise_weights = false,
			for_model_selection=true)

		if particle_accepted
			total_n_accepted += 1
			tracker.model_trackers[m].n_accepted[end] += 1
			tracker.model_trackers[m].population[end] = vcat(tracker.model_trackers[m].population[end], abcsmc_tracker.population[end][1,:]')
			push!(tracker.model_trackers[m].distances[end], abcsmc_tracker.distances[end][1])
			push!(tracker.model_trackers[m].weights[end].values, abcsmc_tracker.weights[end].values[1])
		end

		n_iterations += 1

	end

	println("Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles")
	println("Number of accepted parameters: ", join([string("Model ", m, ": ",  tracker.model_trackers[m].n_accepted[end]) for m in 1:tracker.M], "\t"))

	if n_iterations == tracker.max_iter+1
		warn("Simulated model selection reached maximum number of iterations ($(tracker.max_iter)) on the an SMC run - consider trying more iterations.")
	end

	# Normalise weights for subsequent ABC-SMC run
	for mtracker in tracker.model_trackers
		if size(mtracker.weights[end],1) > 0
			mtracker.weights[end] = normalise(mtracker.weights[end], tosum=1.0)
		end
	end

end

"""
	model_selection(input::SimulatedModelSelectionInput,
		reference_data::AbstractArray{Float64,2})

# Arguments
- `input::SimulatedModelSelectionInput`: A ['SimulatedModelSelectionInput']@(ref) object that contains the settings for the model selection algorithm.
- `reference_data::AbstractArray{Float64,2}`: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)
"""
function model_selection(input::SimulatedModelSelectionInput,
	reference_data::AbstractArray{Float64,2})

	println("Simulated model selection\nPopulation 1")
	tracker = initialise_modelselection(input, reference_data)

	for i in 2:length(input.threshold_schedule)
		println("Population $i")
		iterate_modelselection!(tracker, input.threshold_schedule[i], reference_data)

		# Avoid infinite loop if no particles are accepted
		if sum([tracker.model_trackers[m].n_accepted[end] for m = 1:tracker.M]) == 0
			warn("No particles were accepted in population $i - terminating model selection algorithm")
			break
		end
	end

	return build_modelselection_output(tracker)
end

function build_modelselection_output(tracker::SimulatedModelSelectionTracker)
	return ModelSelectionOutput(
		tracker.M,
		[[tracker.model_trackers[m].n_accepted[i] for m = 1:tracker.M] for i in 1:length(tracker.threshold_schedule)],
		tracker.threshold_schedule,
		[SimulatedABCSMCOutput(
			tracker.model_trackers[m].n_params,
			tracker.model_trackers[m].n_accepted,
			tracker.model_trackers[m].n_tries,
			tracker.threshold_schedule,
			tracker.model_trackers[m].population,
			tracker.model_trackers[m].distances,
			tracker.model_trackers[m].weights)
				for m in 1:tracker.M]
		)
end

function build_modelselection_output(tracker::EmulatedModelSelectionTracker)
	return ModelSelectionOutput(
		tracker.M,
		[[tracker.model_trackers[m].n_accepted[i] for m = 1:tracker.M] for i in 1:length(tracker.threshold_schedule)],
		tracker.threshold_schedule,
		[EmulatedABCSMCOutput(
			tracker.model_trackers[m].n_params,
			tracker.model_trackers[m].n_accepted,
			tracker.model_trackers[m].n_tries,
			tracker.threshold_schedule,
			tracker.model_trackers[m].population,
			tracker.model_trackers[m].distances,
			tracker.model_trackers[m].weights,
			tracker.model_trackers[m].emulators)
				for m in 1:tracker.M]
		)
end

# Initialises the model selection run and runs the first (rejection) population
function initialise_modelselection(input::EmulatedModelSelectionInput, reference_data::AbstractArray{Float64,2})

	#
	# Check input sizes
	#
	if span(input.model_prior) != input.M
		throw(ArgumentError("There are $(input.M) models but the span of the model prior support is $(span(input.model_prior))"))
	end

	if length(input.emulation_settings_arr) != input.M
		throw(ArgumentError("There are $(input.M) models but $(length(input.emulation_settings_arr)) emulation settings"))
	end

	if length(input.parameter_priors) != input.M
		throw(ArgumentError("There are $(input.M) models but $(length(input.parameter_priors)) sets of parameter priors"))
	end

	total_n_accepted = 0
	n_iterations = 1

	#
	# Initialise arrays that will track rejection ABC run for each model - these
	# will be used to create CandidateModelTrackers after the rejection ABC run
	#
	rejection_trackers = [ModelSelectionRejectionTracker(length(input.parameter_priors[m]))
								for m in 1:input.M]

	#
	# Train the emulators
	#
	prior_samplers = [f(n_design_points) = generate_parameters(input.parameter_priors[m], n_design_points)[1] for m in 1:input.M]
	emulators = [input.emulation_settings_arr[m].train_emulator_function(prior_samplers[m]) for m in 1:input.M]

	#
	# Compute first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter
		sampled_models = rand(input.model_prior, min(input.max_batch_size, input.n_particles-total_n_accepted))
		tries_this_it = [size(sampled_models[sampled_models.==m],1) for m=1:input.M]


		[rejection_trackers[m].n_tries += tries_this_it[m] for m=1:input.M]

		for m=1:input.M
			if tries_this_it[m] > 0
				#println("Running rejection batch of size $(tries_this_it[m]) for model $m")
				out = ABCrejection(
						EmulatedABCRejectionInput(length(input.parameter_priors[m]),
											tries_this_it[m],
											input.threshold_schedule[1],
											input.parameter_priors[m],
											input.emulation_settings_arr[m],
											tries_this_it[m],
											1),
				reference_data,
				emulator = emulators[m],
				normalise_weights = false,
				for_model_selection = true)

				# If particle accepted
				if size(out.population,1) >= 1
					#println("Accepted $(size(out.population,1)) particles for model $m")
					total_n_accepted += out.n_accepted
					rejection_trackers[m].n_accepted += out.n_accepted
					rejection_trackers[m].population = vcat(rejection_trackers[m].population, out.population)
					rejection_trackers[m].distances = vcat(rejection_trackers[m].distances, out.distances)
					rejection_trackers[m].weight_values = vcat(rejection_trackers[m].weight_values, out.weights.values)
				elseif size(out.population, 1) > tries_this_it[m]
					error("$(size(out.population,1)) particles were accepted when model $m was only sampled $(tries_this_it[m]) times!")
				end
			end
		end

		n_iterations += 1
	end

	println("Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total")

	if n_iterations == input.max_iter+1
		warn("Emulated model selection reached maximum number of iterations ($(input.max_iter)) on the first population - consider trying more iterations.")
	end

	# Normalise weights for each model
	for m=1:input.M
		rejection_trackers[m].weight_values = rejection_trackers[m].weight_values ./ sum(rejection_trackers[m].weight_values)
	end

	cmTrackers = [EmulatedCandidateModelTracker(
		length(input.parameter_priors[m]),
		[rejection_trackers[m].n_accepted],
		[rejection_trackers[m].n_tries],
		[rejection_trackers[m].population],
		[rejection_trackers[m].distances],
		[StatsBase.Weights(rejection_trackers[m].weight_values, 1.0)],
		input.parameter_priors[m],
		input.emulation_settings_arr[m],
		[emulators[m]])
			for m = 1:input.M]

	println("Number of accepted parameters: ", join([string("Model ", m, ": ",  cmTrackers[m].n_accepted[end]) for m in 1:input.M], "\t"))

	return EmulatedModelSelectionTracker(
				input.M,
				input.n_particles,
				[input.threshold_schedule[1]],
				input.model_prior,
				cmTrackers,
				input.max_batch_size,
				input.max_iter)
end

function model_selection(input::EmulatedModelSelectionInput,
	reference_data::AbstractArray{Float64,2})

	println("Emulated model selection\nPopulation 1 - ABC Rejection ϵ = $(input.threshold_schedule[1])")
	tracker = initialise_modelselection(input, reference_data)

	for i in 2:length(input.threshold_schedule)
		println("Population $i - ABCSMC ϵ = $(input.threshold_schedule[i])")
		iterate_modelselection!(tracker, input.threshold_schedule[i], reference_data)

		# Avoid infinite loop if no particles are accepted for all models
		if sum([tracker.model_trackers[m].n_accepted[end] for m = 1:tracker.M]) == 0
			warn("No particles were accepted in population $i - terminating model selection algorithm")
			break
		end
	end

	return build_modelselection_output(tracker)
end

function iterate_modelselection!(tracker::EmulatedModelSelectionTracker, threshold::Float64,
									reference_data::AbstractArray{Float64,2})

	# Retrain emulators for non-dead models
	# for mtracker in tracker.model_trackers
	# 	if mtracker.n_accepted[end] > 0
	# 		prior_sampling_function = function(n_design_points)
	#             ret_idx = StatsBase.sample(indices(mtracker.population[end], 1), mtracker.weights[end], n_design_points)
	#             return mtracker.population[end][ret_idx, :]
	#         end
	#         emulator = mtracker.emulation_settings.train_emulator_function(prior_sampling_function)
	# 		push!(mtracker.emulators, emulator)
	# 	else
	# 		push!(mtracker.emulators, nothing)
	# 	end
	# end

	# println("Trained emulators")

	# Initialise
	push!(tracker.threshold_schedule, threshold)

	for mtracker in tracker.model_trackers
		push!(mtracker.n_accepted, 0)
		push!(mtracker.n_tries, 0)
		push!(mtracker.population, zeros(0, mtracker.n_params))
		push!(mtracker.distances, zeros(0))
		push!(mtracker.weights, StatsBase.Weights(zeros(0)))
	end

	total_n_accepted = 0
	n_iterations = 1

	while total_n_accepted < tracker.n_particles && n_iterations <= tracker.max_iter
		sampled_models = rand(tracker.model_prior, min(tracker.max_batch_size, tracker.n_particles-total_n_accepted))
		tries_this_it = [size(sampled_models[sampled_models.==m],1) for m=1:tracker.M]

		for m = 1:tracker.M
			# Skip a model that wasn't sampled in this iteration or had no accepted particles in the previous population
			if tries_this_it[m] < 0 || tracker.model_trackers[m].n_accepted[end-1] == 0
				#println("Skipping model $m")
				continue
			end

			smc_tracker = EmulatedABCSMCTracker(
					tracker.model_trackers[m].n_params,
					deepcopy(tracker.model_trackers[m].n_accepted[1:end-1]),
					[0],
					deepcopy(tracker.threshold_schedule[1:end-1]),
					deepcopy(tracker.model_trackers[m].population[1:end-1]),
					deepcopy(tracker.model_trackers[m].distances[1:end-1]),
					deepcopy(tracker.model_trackers[m].weights[1:end-1]),
					tracker.model_trackers[m].priors,
					tracker.model_trackers[m].emulation_settings,
					tries_this_it[m],
					1,
					deepcopy(tracker.model_trackers[m].emulators[1:end-1]))

			particles_accepted = iterateABCSMC!(
									smc_tracker,
									threshold,
									tries_this_it[m],
									reference_data,
									emulator = tracker.model_trackers[m].emulators[end],
									normalise_weights = false,
									for_model_selection = true)

			if particles_accepted
				total_n_accepted += smc_tracker.n_accepted[end]
				tracker.model_trackers[m].n_accepted[end] += smc_tracker.n_accepted[end]
				tracker.model_trackers[m].population[end] = vcat(tracker.model_trackers[m].population[end], smc_tracker.population[end])
				tracker.model_trackers[m].distances[end] = vcat(tracker.model_trackers[m].distances[end], smc_tracker.distances[end])
				tracker.model_trackers[m].weights[end].values = vcat(tracker.model_trackers[m].weights[end], smc_tracker.weights[end])
			end


		end

		n_iterations += 1

	end

	println("Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles")
	println("Number of accepted parameters: ", join([string("Model ", m, ": ",  tracker.model_trackers[m].n_accepted[end]) for m in 1:tracker.M], "\t"))

	if n_iterations == tracker.max_iter+1
		warn("Emulated model selection reached maximum number of iterations ($(tracker.max_iter)) on the an SMC run - consider trying more iterations.")
	end

	# Normalise weights for subsequent ABC-SMC run
	for mtracker in tracker.model_trackers
		if size(mtracker.weights[end],1) > 0
			mtracker.weights[end] = normalise(mtracker.weights[end], tosum=1.0)
		end
	end

end
