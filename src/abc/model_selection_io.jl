abstract type ModelSelectionInput end

"""
	SimulatedModelSelectionInput

An object that defines settings for a simulation-based model selection computation.

# Fields
- `M::Int64`: The number of models.
- `n_particles::Int64`: The number of particles to be accepted per population (at the model level)
- `threshold_schedule::AbstractArray{Float64,1}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.
- `model_prior::DiscreteUnivariateDistribution`: The prior from which models will be sampled.
- `parameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution,1},1}`: Parameter priors for each model. Each element is an array of priors for the corresponding model (one prior per parameter).
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays.
- `simulator_functions::AbstractArray{Function,1}`: Each element is a function that takes a parameter vector as an argument and outputs model results for a single model.
- `max_iter::Integer`: The maximum number of iterations in each population before algorithm termination.
"""
struct SimulatedModelSelectionInput <: ModelSelectionInput
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

struct EmulatedModelSelectionInput <: ModelSelectionInput
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	parameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution,1},1}
	emulator_trainers::AbstractArray{Function, 1}
	max_batch_size::Int64
	max_iter::Integer
end

mutable struct CandidateModelTracker
	n_accepted::Int64
	n_tries::Int64
	population::AbstractArray{Float64,2}
	distances::AbstractArray{Float64,1}
	weight_values::AbstractArray{Float64,1}
end

function CandidateModelTracker(n_params::Int64)
	return CandidateModelTracker(0, 0, zeros(0, n_params), zeros(0), zeros(0))
end

abstract type ModelSelectionTracker end

mutable struct SimulatedModelSelectionTracker <: ModelSelectionTracker
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	smc_trackers::AbstractArray{SimulatedABCSMCTracker,1}
	summary_statistic::Union{String,AbstractArray{String,1},Function}
	distance_function::Function
	max_iter::Integer
end

mutable struct EmulatedModelSelectionTracker <: ModelSelectionTracker
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	smc_trackers::AbstractArray{EmulatedABCSMCTracker,1}
	max_batch_size::Int64
	max_iter::Integer
end

"""
	ModelSelectionOutput

Contains results of a model selection computation, including which models are best supported by the observed data and the parameter poseteriors at each population for each model.

# Fields
- `M::Int64`: The number of models.
- `n_accepted::AbstractArray{AbstractArray{Int64,1},1}`: The number of parameters accepted by each model at each population. `n_accepted[i][j]` contains the number of acceptances for model `j` at population `i`.
- `threshold_schedule::AbstractArray{Float64,1}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.
- `smc_outputs::AbstractArray{ABCSMCOutput,1}`: A ['SimulatedABCSMCOutput']@(ref) or ['EmulatedABCSMCOutput']@(ref) for each model. Use to find details of the ABC results at each population.
- `completed_all_populations::Bool`: Indicates whether the algorithm completed all the populations successfully. A successful population is one where at least one model accepts at least one particle.
"""
struct ModelSelectionOutput
	M::Int64
	n_accepted::AbstractArray{AbstractArray{Int64,1},1}
	threshold_schedule::AbstractArray{Float64,1}
	smc_outputs::AbstractArray{ABCSMCOutput,1}
	completed_all_populations::Bool
end
