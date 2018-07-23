abstract type AbstractModelSelectionInput end

struct SimulatedModelSelectionInput <: AbstractModelSelectionInput
	M::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	parameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution,1},1}
    n_particles::Int64
    summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
    simulator_functions::AbstractArray{Function,1}
end

mutable struct ModelSelectionTracker
	M::Int64
	model_prior::DiscreteUnivariateDistribution
	abcsmc_inputs::AbstractArray{ABCSMCInput}
	abcsmc_trackers::AbstractArray{Union{ABCSMCTracker,Void},1}
	n_particles::Int64
end

mutable struct CandidateModelTracker
    n_accepted::AbstractArray{Int64,1}
	population::AbstractArray{AbstractArray{Float64,2},1}
    distances::AbstractArray{AbstractArray{Float64,1},1}
    weights::AbstractArray{StatsBase.Weights,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    simulator_function::Function
end

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

	n_populations = length(input.threshold_schedule)
	n_accepted = 0
	n_iterations = 1


	#
	# Do first population using rejection-ABC
	#
	while n_accepted < input.n_particles
		println("iteration number $n_iterations")
		m = rand(input.model_prior)
		println("Sampled model $m")

		out = ABCrejection(
				SimulatedABCRejectionInput(length(input.parameter_priors[m]),
											1,
											input.threshold_schedule[1],
											input.parameter_priors[m],
											input.summary_statistic,
											input.distance_function,
											input.simulator_functions[m],
											1),
				reference_data)

		println(out)
		println()

		n_iterations += 1
	end







	# cmTrackers = Array{CandidateModelTracker,1}(input.M)

	# for m in 1:input.M
	# 	cmTrackers[m] = CandidateModelTracker(zeros(n_populations),
	# 		[zeros(0,0) for i in 1:n_populations],
	# 		[zeros(0) for i in 1:n_populations],
	# 		)
	# end

	# abcsmc_inputs = Array{ABCSMCInput}(input.M)

	# for m in 1:input.M
	# 	abcsmc_inputs[m] = SimulatedABCSMCInput(length(input.parameter_priors[m]),
	# 		1,
	# 		input.threshold_schedule,
	# 		input.parameter_priors[m],
	# 		input.summary_statistic,
	# 		input.distance_function,
	# 		input.simulator_functions[m])
	# end

	# return ModelSelectionTracker(input.M,
	# 	input.model_prior,
	# 	abcsmc_inputs,
	# 	[nothing for m in 1:input.M],
	# 	input.n_particles)
end	

function ABCSMCModelSelection(input::SimulatedModelSelectionInput)
	return 0
end
