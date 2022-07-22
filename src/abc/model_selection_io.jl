abstract type ModelSelectionInput end

struct SimulatedModelSelectionInput{AF<:AbstractFloat, CUD<:ContinuousUnivariateDistribution} <: ModelSelectionInput
	M::Int
	n_particles::Int
	threshold_schedule::AbstractArray{AF,1}
	model_prior::DiscreteUnivariateDistribution
	parameter_priors::AbstractArray{Array{CUD, 1},1}
	distance_simulation_input::AbstractArray{DistanceSimulationInput,1}
	max_iter::Int
end

struct EmulatedModelSelectionInput{AF<:AbstractFloat, CUD<:ContinuousUnivariateDistribution,
	    ER<:AbstractEmulatorRetraining, EPS<:AbstractEmulatedParticleSelection, ET<:AbstractEmulatorTraining} <: ModelSelectionInput
	M::Int
	n_particles::Int
	threshold_schedule::AbstractArray{AF,1}
	model_prior::DiscreteUnivariateDistribution
	parameter_priors::AbstractArray{Array{CUD,1},1}
	emulator_training_input::AbstractArray{EmulatorTrainingInput{ET},1}
	emulator_retraining::ER
	emulated_particle_selection::EPS
	max_batch_size::Int
	max_iter::Int
end

mutable struct CandidateModelTracker{AF<:AbstractFloat}
	n_accepted::Int
	n_tries::Int
	population::AbstractArray{AF,2}
	distances::AbstractArray{AF,1}
	weight_values::AbstractArray{AF,1}
end

function CandidateModelTracker(n_params::Int)
	return CandidateModelTracker(0, 0, zeros(0, n_params), zeros(0), zeros(0))
end

abstract type ModelSelectionTracker end

mutable struct SimulatedModelSelectionTracker <: ModelSelectionTracker
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	smc_trackers::AbstractArray{SimulatedABCSMCTracker,1}
	max_iter::Int
end

mutable struct EmulatedModelSelectionTracker{CUD<:ContinuousUnivariateDistribution, ET,
        ER<:AbstractEmulatorRetraining, EPS<:AbstractEmulatedParticleSelection} <: ModelSelectionTracker
	M::Int64
	n_particles::Int64
	threshold_schedule::AbstractArray{Float64,1}
	model_prior::DiscreteUnivariateDistribution
	smc_trackers::AbstractArray{EmulatedABCSMCTracker{CUD, ET, ER, EPS},1}
	max_batch_size::Int64
	max_iter::Int
end

"""
	ModelSelectionOutput

Contains results of a model selection computation, including which models are best supported by the observed data and the parameter posteriors at each population for each model.

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
