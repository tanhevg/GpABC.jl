module GpABC

export
	AbstractGPKernel,
	SquaredExponentialIsoKernel, SquaredExponentialArdKernel,
	MaternIsoKernel, MaternArdKernel, ExponentialIsoKernel, ExponentialArdKernel,

	AbstractGaussianProcess,
	GPModel,
    MvUniform,
    LatinHypercubeSampler,
    gp_loglikelihood, gp_loglikelihood_log, gp_loglikelihood_grad,
	gp_regression, gp_regression_sample,
	covariance, covariance_training, covariance_diagonal, covariance_grad,
	scaled_squared_distance, scaled_squared_distance_grad,
	get_hyperparameters_size, set_hyperparameters,
    gp_train,

	ABCRejectionOutput, ABCSMCOutput,
	SimulatedABCRejection, SimulatedABCSMC,
	EmulatedABCRejection, EmulatedABCSMC,
	read_rejection_output, read_smc_output,

	LNAInput, LNA, compute_LNA, sample_LNA_trajectories, get_LNA_trajectories,

	SimulatedModelSelection, EmulatedModelSelection, ModelSelectionOutput,
    AbstractEmulatorTraining, DefaultEmulatorTraining,
    EmulatorTrainingInput,
    AbstractEmulatorRetraining, NoopRetraining, IncrementalRetraining, PreviousPopulationRetraining, PreviousPopulationThresholdRetraining,
    AbstractEmulatedParticleSelection, MeanEmulatedParticleSelection, MeanVarEmulatedParticleSelection;

using Optim, Distributions, Distances, OrdinaryDiffEq, ForwardDiff, LinearAlgebra, Logging

import StatsBase

import Base: write

include("gp/kernels/scaled_squared_distance.jl")
include("gp/kernels/abstract_kernel.jl")
include("gp/kernels/rbf_kernels.jl")
include("gp/kernels/matern_kernels.jl")
include("gp/gp.jl")
include("gp/gp_optimisation.jl")

include("abc/io.jl")
include("abc/summary_stats.jl")
include("abc/rejection.jl")
include("abc/smc.jl")
include("abc/model_selection_io.jl")
include("abc/model_selection.jl")
include("abc/simulation.jl")
include("abc/emulation.jl")
include("abc/plot_recipe.jl")
include("util/lna.jl")
include("util/emulation_helpers.jl")

end;
