abstract type AbstractModelSelectionInput end

struct SimulatedModelSelectionInput <: AbstractModelSelectionInput
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	parameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution,1},1}
    summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
    simulator_functions::AbstractArray{Function,1}
    max_iter::Integer
end

mutable struct CandidateModelTracker
	n_params::Integer
    n_accepted::AbstractArray{Int64,1}
    n_tries::AbstractArray{Int64,1}
	population::AbstractArray{AbstractArray{Float64,2},1}
    distances::AbstractArray{AbstractArray{Float64,1},1}
    weights::AbstractArray{StatsBase.Weights,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    simulator_function::Function
end

abstract type ModelSelectionTracker end

mutable struct SimulatedModelSelectionTracker <: ModelSelectionTracker
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	model_trackers::AbstractArray{CandidateModelTracker,1}
	summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
	max_iter::Integer
end

abstract type ModelSelectionOutput end

struct SimulatedModelSelectionOutput <: ModelSelectionOutput
	M::Int64
	n_accepted::AbstractArray{AbstractArray{Int64,1},1}
	threshold_schedule::AbstractArray{Float64,1}
	smc_outputs::AbstractArray{SimulatedABCSMCOutput,1}
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
	# will be used to create CandidateModelTrackers after the rejection ABC run
	# n_accepted(model), n_tries, parameters, distances, weights
	#
	rejection_trackers = [[0, 0, zeros(0,length(input.parameter_priors[m])), Array{Float64,1}(), Array{Float64,1}()]
		for m in 1:input.M]

	#
	# Compute first population using rejection-ABC
	#
	while total_n_accepted < input.n_particles && n_iterations <= input.max_iter
		m = rand(input.model_prior)
		rejection_trackers[m][2] += 1

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
				hide_maxiter_warning=true)

		# If particle accepted
		if size(out.population,1) == 1
			total_n_accepted += 1
			rejection_trackers[m][1] += 1
			rejection_trackers[m][3] = vcat(rejection_trackers[m][3], out.population)
			push!(rejection_trackers[m][4], out.distances[1])
			push!(rejection_trackers[m][5], out.weights.values[1])
		elseif size(out.population, 1) > 1
			error("$(size(out.population,1)) particles were accepted!")
		end

		n_iterations += 1
	end

	println("Completed $(n_iterations-1) iterations, accepting $total_n_accepted particles in total")

	if n_iterations == input.max_iter+1
		warn("Simulated model selection reached maximum number of iterations ($(input.max_iter)) on the first population - consider trying more iterations.")
	end

	cmTrackers = [CandidateModelTracker(
		length(input.parameter_priors[m]),
		[rejection_trackers[m][1]],
		[rejection_trackers[m][2]],
		[rejection_trackers[m][3]],
		[rejection_trackers[m][4]],
		[StatsBase.Weights(rejection_trackers[m][5], 1.0)],
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

		abcsmc_tracker = iterateABCSMC!(
			SimulatedABCSMCTracker(
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
				1),
			threshold,
			1,
			reference_data,
			normalise_weights = false,
			hide_maxiter_warning=true)

		if abcsmc_tracker.n_accepted[end] == 1
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
			mtracker.weights[end] = deepcopy(normalise(mtracker.weights[end], tosum=1.0))
		end
	end

end

function model_selection(input::SimulatedModelSelectionInput,
	reference_data::AbstractArray{Float64,2})

	println("Population 1")
	tracker = initialise_modelselection(input, reference_data)

	for i in 2:length(input.threshold_schedule)
		println("Population $i")
		iterate_modelselection!(tracker, input.threshold_schedule[i], reference_data)

		# Avoid infinite loop if no particles are accepted
		if sum([tracker.model_trackers[m].n_accepted[end] for m = 1:tracker.M]) == 0
			warn("No particles were accepted in population $i - exiting")
			break
		end
	end

	return build_modelselection_output(tracker)
end

function build_modelselection_output(tracker::SimulatedModelSelectionTracker)
	return SimulatedModelSelectionOutput(
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