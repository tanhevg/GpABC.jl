var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#GpAbc.jl-1",
    "page": "Home",
    "title": "GpAbc.jl",
    "category": "section",
    "text": "A Julia package for model parameter estimation with Approximate Bayesian Computation (ABC), using emulation with Gaussian Process regression (GPR)."
},

{
    "location": "#Use-cases-1",
    "page": "Home",
    "title": "Use cases",
    "category": "section",
    "text": "Run Gaussian Process regression\nEstimate model parameters using Rejection ABC without emulation (simulation only)\nEstimate model parameters using Rejection ABC with GPR emulation\nEstimate model parameters using Sequential Monte Carlo (SMC) ABC without emulation (simulation only)\nEstimate model parameters using ABC-SMC with GPR emulationExamples"
},

{
    "location": "examples/#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "examples/#Examples-1",
    "page": "Examples",
    "title": "Examples",
    "category": "section",
    "text": "TODO"
},

{
    "location": "reference/#GpAbc.AbstractGPKernel",
    "page": "Reference",
    "title": "GpAbc.AbstractGPKernel",
    "category": "type",
    "text": "AbstractGPKernel\n\nAbstract kernel type. User-defined kernels should derive from it.\n\nImplementations have to provide methods for get_hyperparameters_size and covariance. Methods for covariance_training, covariance_diagonal and covariance_grad are optional.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.GPModel",
    "page": "Reference",
    "title": "GpAbc.GPModel",
    "category": "type",
    "text": "GPModel\n\nThe main type that is used by most functions withing the package.\n\nAll data matrices are row-major.\n\nFields\n\nkernel::AbstractGPKernel: the kernel\ngp_training_x::AbstractArray{Float64, 2}: training x. Size: n times d.\ngp_training_y::AbstractArray{Float64, 2}: training y. Size: n times 1.\ngp_test_x::AbstractArray{Float64, 2}: test x.  Size: m times d.\ngp_hyperparameters::AbstractArray{Float64, 1}: kernel hyperparameters, followed by standard deviation of intrinsic noise sigma_n, which is always the last element in the array.\ncache::HPOptimisationCache: cache of matrices that can be re-used between calls to gp_loglikelihood and gp_loglikelihood_grad\n\n\n\n"
},

{
    "location": "reference/#GpAbc.GPModel",
    "page": "Reference",
    "title": "GpAbc.GPModel",
    "category": "type",
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        kernel::AbstractGPKernel\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nConstructor of GPModel that allows the kernel to be specified. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.GPModel",
    "page": "Reference",
    "title": "GpAbc.GPModel",
    "category": "type",
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nDefault constructor of GPModel, that will use SquaredExponentialIsoKernel. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.GPModel-Tuple{}",
    "page": "Reference",
    "title": "GpAbc.GPModel",
    "category": "method",
    "text": "GPModel(;training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    kernel::AbstractGPKernel=SquaredExponentialIsoKernel(),\n    gp_hyperparameters::AbstractArray{Float64, 1}=Array{Float64}(0))\n\nConstructor of GPModel with explicit arguments. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.MaternArdKernel",
    "page": "Reference",
    "title": "GpAbc.MaternArdKernel",
    "category": "type",
    "text": "MaternArdKernel <: AbstractGPKernel\n\nMatérn kernel with distinct length scale for each dimention, l_k. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n"
},

{
    "location": "reference/#GpAbc.MaternIsoKernel",
    "page": "Reference",
    "title": "GpAbc.MaternIsoKernel",
    "category": "type",
    "text": "MaternIsoKernel <: AbstractGPKernel\n\nMatérn kernel with uniform length scale across all dimensions, l. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n"
},

{
    "location": "reference/#GpAbc.SquaredExponentialArdKernel",
    "page": "Reference",
    "title": "GpAbc.SquaredExponentialArdKernel",
    "category": "type",
    "text": "SquaredExponentialArdKernel <: AbstractGPKernel\n\nSquared exponential kernel with distinct length scale for each dimention, l_k.\n\nK(r) = sigma_f^2 e^-r2 r_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n"
},

{
    "location": "reference/#GpAbc.SquaredExponentialIsoKernel",
    "page": "Reference",
    "title": "GpAbc.SquaredExponentialIsoKernel",
    "category": "type",
    "text": "SquaredExponentialIsoKernel <: AbstractGPKernel\n\nSquared exponential kernel with uniform length scale across all dimensions, l.\n\nK(r) = sigma_f^2 e^-r2 r_ij = sum_k=1^dfrac(x_ik-z_jk)^2l^2\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n"
},

{
    "location": "reference/#GpAbc.ExponentialArdKernel-Tuple{}",
    "page": "Reference",
    "title": "GpAbc.ExponentialArdKernel",
    "category": "method",
    "text": "ExponentialArdKernel\n\nAlias for MaternArdKernel(1)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.ExponentialIsoKernel-Tuple{}",
    "page": "Reference",
    "title": "GpAbc.ExponentialIsoKernel",
    "category": "method",
    "text": "ExponentialIsoKernel\n\nAlias for MaternIsoKernel(1)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.covariance-Tuple{GpAbc.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.covariance",
    "category": "method",
    "text": "covariance(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nReturn the covariance matrix. Should be overridden by kernel implementations.\n\nArguments\n\nker: The kernel object. Implementations must override with their own subtype.\nlog_theta: natural logarithm of hyperparameters.\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\n\nReturn\n\nThe covariance matrix, of size n times m.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.covariance_diagonal-Tuple{GpAbc.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.covariance_diagonal",
    "category": "method",
    "text": "covariance_diagonal(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2})\n\nThis is a speedup version of covariance, which is invoked if the caller is not interested in the entire covariance matrix, but only needs the variance, i.e. the diagonal of the covariance matrix.\n\nDefault method just returns diag(covariance(...)), with x === z. Kernel implementations can optionally override it to achieve betrer performance, by not computing the non diagonal elements of covariance matrix.\n\nSee covariance for description of arguments.\n\nReturn\n\nThe 1-d array of variances, of size size(x, 1).\n\n\n\n"
},

{
    "location": "reference/#GpAbc.covariance_grad-Tuple{GpAbc.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.covariance_grad",
    "category": "method",
    "text": "covariance_grad(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n\nReturn the gradient of the covariance function with respect to logarigthms of hyperparameters, based on the provided direction matrix.\n\nThis function can be optionally overridden by kernel implementations. If the gradient function is not provided, gp_train will fail back to NelderMead algorithm by default.\n\nArguments\n\nker: The kernel object. Implementations must override with their own subtype.\nlog_theta:  natural logarithm of hyperparameters\nx: Training data, reshaped into a 2-d array. x must have dimensions n times d.\nR the directional matrix, n times n\n\nR = frac1sigma_n^2(alpha * alpha^T - K^-1) alpha = K^-1y\n\nReturn\n\nA vector of size length(log_theta), whose j\'th element is equal to\n\ntr(R fracpartial Kpartial eta_j)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.covariance_training-Tuple{GpAbc.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.covariance_training",
    "category": "method",
    "text": "covariance_training(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    training_x::AbstractArray{Float64, 2})\n\nThis is a speedup version of covariance, which is only called during traing sequence. Intermediate matrices computed in this function for particular hyperparameters can be cached and reused subsequently, either in this function or in covariance_grad\n\nDefault method just delegates to covariance with x === z. Kernel implementations can optionally override it for betrer performance.\n\nSee covariance for description of arguments and return values.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.get_hyperparameters_size-Tuple{GpAbc.AbstractGPKernel,AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.get_hyperparameters_size",
    "category": "method",
    "text": "get_hyperparameters_size(kernel::AbstractGPKernel, training_data::AbstractArray{Float64, 2})\n\nReturn the number of hyperparameters for used by this kernel on this training data set. Should be overridden by kernel implementations.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.gp_loglikelihood-Tuple{AbstractArray{Float64,1},GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_loglikelihood",
    "category": "method",
    "text": "gp_loglikelihood(theta::AbstractArray{Float64, 1}, gpm::GPModel)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.gp_loglikelihood-Tuple{GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_loglikelihood",
    "category": "method",
    "text": "gp_loglikelihood(gpm::GPModel)\n\nCompute the log likelihood function, based on the kernel and training data specified in gpm.\n\nlog p(y vert X theta) = - frac12(y^TK^-1y + log vert K vert + n log 2 pi)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.gp_regression-Tuple{GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_regression",
    "category": "method",
    "text": "gp_regression(gpm::GPModel; <optional keyword arguments>)\n\nRun the Gaussian Process Regression.\n\nArguments\n\ngpm: the GPModel, that contains the training data (x and y), the kernel, the hyperparameters and the test data for running the regression.\ntest_x: if specified, overrides the test x in gpm. Size m times d.\nlog_level::Int (optional): log level. Default is 0, which is no logging at all. 1 makes gp_regression print basic information to standard output.\nfull_covariance_matrix::Bool (optional): whether we need the full covariance matrix, or just the variance vector. Defaults to false (i.e. just the variance).\nbatch_size::Int (optional): If full_covariance_matrix is set to false, then the mean and variance vectors will be computed in batches of this size, to avoid allocating huge matrices. Defaults to 1000.\nobservation_noise::Bool (optional): whether the observation noise (with variance sigma_n^2) should be included in the output variance. Defaults to true.\n\nReturn\n\nA tuple of (mean, var). mean is a mean vector of the output multivariate Normal distribution, of size m. var is either the covariance matrix of size m times m, or a variance vector of size m, depending on full_covariance_matrix flag.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.gp_regression-Tuple{Union{AbstractArray{Float64,1}, AbstractArray{Float64,2}},GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_regression",
    "category": "method",
    "text": "gp_regression(test_x::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}},\n    gpem::GPModel; <optional keyword arguments>)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.gp_train-Tuple{GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_train",
    "category": "method",
    "text": "gp_train(gpm::GPModel; <optional keyword arguments>)\n\nFind Maximum Likelihood Estimate of Gaussian Process hyperparameters by maximising gp_loglikelihood, using Optim package.\n\nArguments\n\ngpm: the GPModel, that contains the training data (x and y), the kernel and the starting hyperparameters that will be used for optimisation.\noptimisation_solver_type::Type{<:Optim.Optimizer} (optional): the solver to use. If not given, then ConjugateGradient will be used for kernels that have gradient implementation, and NelderMead will be used for those that don\'t.\nhp_lower::AbstractArray{Float64, 1} (optional): the lower boundary for box optimisation. Defaults to e^-10 for all hyperparameters.\nhp_upper::AbstractArray{Float64, 1} (optional): the upper boundary for box optimisation. Defaults to e^10 for all hyperparameters.\nlog_level::Int (optional): log level. Default is 0, which is no logging at all. 1 makes gp_train print basic information to standard output. 2 switches Optim logging on, in addition to 1.\n\nReturn\n\nThe list of all hyperparameters, including the standard deviation of the measurement noise sigma_n. Note that after this function returns, the hyperparameters of gpm will be set to the optimised value, and there is no need to call set_hyperparameters once again.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.rejection_abc-Tuple{Function,Function}",
    "page": "Reference",
    "title": "GpAbc.rejection_abc",
    "category": "method",
    "text": "test_prior - A no-argument function, returns a sample from test prior, that is ready to be fed into summary_statistics_function. Returns an array of size (n, d), n rows and d columns, where n is the number of points, and d is the dimentionality of the data\n\nsummary_statistics_function - A function that takes in a test sample, and returns a 1D vector of summary statistics. In our case this returns a vector of norms to the observed data, as emulated by the GP.\n\nobserved_summary_statistic - Summary statistic of the observed data. In our case this is zero\n\nSee abc_test.jl for usage example\n\n\n\n"
},

{
    "location": "reference/#GpAbc.scaled_squared_distance-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.scaled_squared_distance",
    "category": "method",
    "text": "scaled_squared_distance(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nCompute the scaled squared distance between x and z:\n\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\n\nReturn\n\nAn n times m matrix of scaled squared distances\n\n\n\n"
},

{
    "location": "reference/#GpAbc.scaled_squared_distance_grad-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.scaled_squared_distance_grad",
    "category": "method",
    "text": "scaled_squared_distance_grad(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n\nReturn the gradient of the scaled_squared_distance function with respect to logarigthms of length scales, based on the provided direction matrix.\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\nR the direction matrix, n times m. This can be used to compute the gradient of a function that depends on scaled_squared_distance via the chain rule.\n\nReturn\n\nA vector of size length(log_ell), whose k\'th element is equal to\n\ntexttr(R fracpartial Kpartial l_k)\n\n\n\n"
},

{
    "location": "reference/#GpAbc.set_hyperparameters-Tuple{GpAbc.GPModel,AbstractArray{Float64,1}}",
    "page": "Reference",
    "title": "GpAbc.set_hyperparameters",
    "category": "method",
    "text": "set_hyperparameters(gpm::GPModel, hypers::AbstractArray{Float64, 1})\n\nSet the hyperparameters of the GPModel\n\n\n\n"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": "Modules = [GpAbc]"
},

]}
