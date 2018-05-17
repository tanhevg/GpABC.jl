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
	multiple_training_abc,
	multiple_training_seq_abc,
	smc,
	get_head_of_ref_table,
	rejection_abc;

using Optim, Distributions

include("kernels/scaled_squared_distance.jl")
include("kernels/abstract_kernel.jl")
include("kernels/rbf_kernels.jl")
include("kernels/matern_kernels.jl")
include("gp.jl")
include("abc.jl")
include("gp_optimisation.jl")
include("multiple_training_abc.jl")
include("mv_uniform.jl")
include("latin_hypercube_sampler.jl")

end;
