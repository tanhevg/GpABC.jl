module GpAbc

export
	AbstractGPKernel,
	SquaredExponentialIsoKernel, SquaredExponentialArdKernel,
	MaternIsoKernel, MaternArdKernel, ExponentialIsoKernel, ExponentialArdKernel,

	AbstractGaussianProcess,
	GPModel,
    MvUniform,
    LatinHypercubeSampler,
    gp_loglikelihood, gp_loglikelihood_log, gp_loglikelihood_grad, gp_regression,
	covariance, covariance_training, covariance_diagonal, covariance_grad,
	scaled_squared_distance, scaled_squared_distance_grad,
	get_hyperparameters_size, set_hyperparameters,
    gp_train,

	ABCrejection,
	initialiseABCSMC, iterateABCSMC!, ABCSMC,
	ABCInput, ABCOutput,
	ABCRejectionInput, ABCSMCInput,
	SimulatedABCRejectionInput, EmulatedABCRejectionInput,
	SimulatedABCSMCInput, EmulatedABCSMCInput,
	ABCRejectionOutput, ABCSMCOutput,
	ABCSMCTracker,
	SimulatedABCSMCTracker, EmulatedABCSMCTracker,
	read_rejection_output, read_smc_output;

	# multiple_training_abc,
	# multiple_training_seq_abc,
	# smc,
	# get_head_of_ref_table,
	# rejection_abc,

using Optim, Distributions

import StatsBase

import Base: write


include("gp/kernels/scaled_squared_distance.jl")
include("gp/kernels/abstract_kernel.jl")
include("gp/kernels/rbf_kernels.jl")
include("gp/kernels/matern_kernels.jl")
include("gp/gp.jl")
include("gp/gp_optimisation.jl")
include("util/mv_uniform.jl")
include("util/latin_hypercube_sampler.jl")

include("abc/io.jl")
include("abc/rejection.jl")
include("abc/smc.jl")
include("abc/summary_stats.jl")


# include("abc.jl")
# include("multiple_training_abc.jl")

end;
