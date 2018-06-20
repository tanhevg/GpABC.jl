var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "Notation\nBasic Usage\nTraining the GP\nKernelsTODO markdown does not support include, so copy-paste content from /README.md once it is finalised"
},

{
    "location": "#Notation-1",
    "page": "Home",
    "title": "Notation",
    "category": "section",
    "text": "Throughout this manual, we denote the number of training points as n, and the number of test points as m. The number of dimensions is denoted as d. For one-dimensional case, where each individual training and test point is just a real number, both one-dimensional and two-dimensional arrays are accepted as inputs. In Basic Gaussian Process Regression Example training_x can either be a vector of size n, or an n times 1 matrix. For a multidimensional case, where test and training points are elements of a d-dimentional space, all inputs have to be row major, so training_x and test_x become an n times d and an m times d matrices, respectively."
},

{
    "location": "#Basic-Usage-1",
    "page": "Home",
    "title": "Basic Usage",
    "category": "section",
    "text": "The package is built around a type GPModel, which encapsulates all the information required for training the Gaussian Process and performing the regression. In the simplest scenario the user would instantiate this type with some training data and labels, provide the hyperparameters and run the regression. By default, SquaredExponentialIsoKernel will be used. This scenario is illustrated by Basic Gaussian Process Regression Example."
},

{
    "location": "#Training-the-GP-1",
    "page": "Home",
    "title": "Training the GP",
    "category": "section",
    "text": "Normally, kernel hyperparameters are not known in advance. In this scenario the training function gp_train should be used to find the Maximum Likelihood Estimate (MLE) of hyperparameters. This is demonstrated in Optimising Hyperparameters for GP Regression Example.GaussProABC uses Optim package for optimising the hyperparameters. By default, Conjugate Gradient bounded box optimisation is used, as long as the gradient with respect to hyperparameters is implemented for the kernel function. If the gradient implementation is not provided, Nelder Mead optimiser is used by default.The starting point of the optimisation can be specified by calling set_hyperparameters. If the starting point has not been provided, optimisation will start from all hyperparameters set to 1. Default upper and lower bounds are set to e^10 and e^10 , respectively, for each hyperparameter.For numerical stability the package uses logarithms of hyperparameters internally, when calling the log likelihood and kernel functions. Logarithmisation and exponentiation back takes place in gp_train function.The log likelihood function with log hyperparameters is implemented by gp_loglikelihood_log. This is the target function of the optimisation procedure in gp_train. There is also a version of log likelihood with actual (non-log) hyperparameters: gp_loglikelihood. The gradient of the log likelihood function with respect to logged hyperparameters is implemented by gp_loglikelihood_grad.Depending on the kernel, it is not uncommon for the log likelihood function to have multiple local optima. If a trained GP produces an unsatisfactory data fit, one possible workaround is trying to run gp_train several times with random starting points. This approach is demonstrated in Advanced Usage of gp_train example.Optim has a built in constraint of running no more than 1000 iterations of any optimisation algorithm. GpAbc relies on this feature to ensure that the training procedure does not get stuck forever. As a consequence, the optimizer might exit prematurely, before reaching the local optimum. Setting log_level argument of gp_train to a value greater than zero will make it log its actions to standard output, including whether the local minimum has been reached or not."
},

{
    "location": "#Kernels-1",
    "page": "Home",
    "title": "Kernels",
    "category": "section",
    "text": "GpAbc ships with an extensible library of kernel functions. Each kernel is represented with a type that derives from AbstractGPKernel:SquaredExponentialIsoKernel\nSquaredExponentialArdKernel\nMaternIsoKernel\nMaternArdKernel\nExponentialIsoKernel\nExponentialArdKernelThese kernels rely on matrix of scaled squared distances between training/test inputs r_ij, which is computed by scaled_squared_distance function. The gradient vector of scaled squared distance derivatives with respect to length scale hyperparameter(s) is returned by scaled_squared_distance_grad function.The kernel covariance matrix is returned by function covariance. Optional speedups of this function covariance_diagonal and covariance_training are implemented for the pre-shipped kernels. The gradient with respect to log hyperparameters is computed by covariance_grad. The log_theta argument refers to the logarithms of kernel hyperparameters. Note that hyperparameters that do not affect the kernel (e.g. sigma_n ) are not included in log_theta.Custom kernels functions can be implemented  by adding more types that inherit from AbstractGPKernel. This is demonstrated in Using a Custom Kernel Example"
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
    "text": "Basic Gaussian Process Regression\nOptimising Hyperparameters for GP Regression\nAdvanced Usage of gp_train\nUsing a Custom Kernel"
},

{
    "location": "examples/#example-1-1",
    "page": "Examples",
    "title": "Basic Gaussian Process Regression",
    "category": "section",
    "text": "using GpAbc, Distributions, PyPlot\n\n# prepare the data\nn = 30\nf(x) = x.ˆ2 + 10 * sin.(x) # the latent function\n\ntraining_x = sort(rand(Uniform(-10, 10), n))\ntraining_y = f(training_x)\ntraining_y += 20 * (rand(n) - 0.5) # add some noise\ntest_x = collect(linspace(min(training_x...), max(training_x...), 1000))\n\n # SquaredExponentialIsoKernel is used by default\ngpm = GPModel(training_x, training_y)\n\n# pretend we know the hyperparameters in advance\n# σ_f = 37.08; l = 1.0; σ_n = 6.58. See SquaredExponentialIsoKernel documentation for details\nset_hyperparameters(gpm, [37.08, 1.0, 6.58])\n(test_y, test_var) = gp_regression(test_x, gpm)\n\nplot(test_x, [test_y f(test)]) # ... and more sophisticated plotting"
},

{
    "location": "examples/#example-2-1",
    "page": "Examples",
    "title": "Optimising Hyperparameters for GP Regression",
    "category": "section",
    "text": "Based on Basic Gaussian Process Regression, but with added optimisation of hyperparameters:using GaussProABC\n\n# prepare the data ...\n\ngpm = GPModel(training_x, training_y)\n\n # by default, the optimiser will start with all hyperparameters set to 1,\n # constrained between exp(-10) and exp(10)\ntheta_mle = gp_train(gpm)\n\n# optimised hyperparameters are stored in gpm, so no need to pass them again\n(test_y, test_var) = gp_regression(test_x, gpm)"
},

{
    "location": "examples/#example-3-1",
    "page": "Examples",
    "title": "Advanced Usage of gp_train",
    "category": "section",
    "text": "using GpAbc, Optim, Distributions\n\nfunction gp_train_advanced(gpm::GPModel, attempts::Int)\n    # Initialise the bounds, with special treatment for the second hyperparameter\n    p = get_hyperparameters_size(gpm)\n    bound_template = ones(p)\n    upper_bound = bound_template * 10\n    upper_bound[2] = 2\n    lower_bound = bound_template * -10\n    lower_bound[2] = -1\n\n    # Starting point will be sampled from a Multivariate Uniform distribution\n    start_point_distr = MvUniform(lower_bound, upper_bound)\n\n    # Run several attempts of training and store the\n    # minimiser hyperparameters and the value of log likelihood function\n    hypers = Array{Float64}(attempts, p)\n    likelihood_values = Array{Float64}(attempts)\n    for i=1:attempts\n        set_hyperparameters(gpm, exp.(rand(start_point_distr)))\n        hypers[i, :] = gp_train(gpm,\n            optimisation_solver_type=SimulatedAnnealing, # note the solver type\n            hp_lower=lower_bound, hp_upper=upper_bound, log_level=1)\n        likelihood_values[i] = gp_loglikelihood(gpm)\n    end\n    # Retain the hyperparameters where the maximum log likelihood function is attained\n    gpm.gp_hyperparameters = hypers[indmax(likelihood_values), :]\nend"
},

{
    "location": "examples/#example-4-1",
    "page": "Examples",
    "title": "Using a Custom Kernel",
    "category": "section",
    "text": "The methods below should be implemented for the custom kernel, unless indicated as optional. Please see reference documentation for detailed description of each method and parameter.using GpAbc\nimport GpAbc.covariance, GpAbc.get_hyperparameters_size\n\n\"\"\"\n   This is the new kernel that we are adding\n\"\"\"\ntype MyCustomkernel <: AbstractGPKernel\n\n    # optional cache of matrices that could be re-used between calls to\n    # covariance_training and covariance_grad, keyed by hyperparameters\n    cache::MyCustomCache\nend\n\n\"\"\"\n    Report the number of hyperparameters required by the new kernel\n\"\"\"\nfunction get_hyperparameters_size(ker::MyCustomkernel, training_data::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n    Covariance function of the new kernel.\n\n    Return the covariance matrix. Assuming x is an n by d matrix, and z is an m by d matrix,\n    this should return an n by m matrix. Use `scaled_squared_distance` helper function here.\n\"\"\"\nfunction covariance(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n    Optional speedup of `covariance` function, that is invoked when the calling code is\n    only interested in variance (i.e. diagonal elements of the covariance) of the kernel.\n\"\"\"\nfunction covariance_diagonal(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n   Optional speedup of `covariance` function that is invoked during training of the GP.\n   Intermediate matrices that are re-used between this function and `covariance_grad` could\n   be cached in `ker.cache`\n\"\"\"\nfunction covariance_training(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    training_x::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n    Optional gradient of `covariance` function with respect to hyperparameters, required\n    for optimising with `ConjugateGradient` method. If not provided, `NelderMead` optimiser\n    will be used.\n\n    Use `scaled_squared_distance_grad` helper function here.\n\"\"\"\nfunction covariance_grad(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n    # ...\nend\n\ngpm = GPModel(training_x, training_y, MyCustomkernel())\ntheta_mle = gp_train(gpm)\n(test_y, test_var) = gp_regression(test_x, gpm)"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Index-1",
    "page": "Reference",
    "title": "Index",
    "category": "section",
    "text": "Modules = [GpAbc]"
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
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        kernel::AbstractGPKernel\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nConstructor of GPModel that allows the kernel to be specified. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
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
    "text": "MaternArdKernel <: AbstractGPKernel\n\nMatérn kernel with distinct length scale for each dimention, l_k. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n"
},

{
    "location": "reference/#GpAbc.MaternIsoKernel",
    "page": "Reference",
    "title": "GpAbc.MaternIsoKernel",
    "category": "type",
    "text": "MaternIsoKernel <: AbstractGPKernel\n\nMatérn kernel with uniform length scale across all dimensions, l. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n"
},

{
    "location": "reference/#GpAbc.SquaredExponentialArdKernel",
    "page": "Reference",
    "title": "GpAbc.SquaredExponentialArdKernel",
    "category": "type",
    "text": "SquaredExponentialArdKernel <: AbstractGPKernel\n\nSquared exponential kernel with distinct length scale for each dimention, l_k.\n\nbeginaligned\nK(r)  = sigma_f^2 e^-r2 \nr_ij  = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n"
},

{
    "location": "reference/#GpAbc.SquaredExponentialIsoKernel",
    "page": "Reference",
    "title": "GpAbc.SquaredExponentialIsoKernel",
    "category": "type",
    "text": "SquaredExponentialIsoKernel <: AbstractGPKernel\n\nSquared exponential kernel with uniform length scale across all dimensions, l.\n\nbeginaligned\nK(r)  = sigma_f^2 e^-r2 \nr_ij  = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n"
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
    "location": "reference/#GpAbc.gp_loglikelihood_grad-Tuple{AbstractArray{Float64,1},GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_loglikelihood_grad",
    "category": "method",
    "text": "gp_loglikelihood_grad(theta::AbstractArray{Float64, 1}, gpem::GPModel)\n\nGradient of the log likelihood function (gp_loglikelihood_log) with respect to logged hyperparameters.\n\n\n\n"
},

{
    "location": "reference/#GpAbc.gp_loglikelihood_log-Tuple{AbstractArray{Float64,1},GpAbc.GPModel}",
    "page": "Reference",
    "title": "GpAbc.gp_loglikelihood_log",
    "category": "method",
    "text": "gp_loglikelihood_log(theta::AbstractArray{Float64, 1}, gpm::GPModel)\n\nLog likelihood function with log hyperparameters. This is the target function of the hyperparameters optimisation procedure. Its gradient is coputed by gp_loglikelihood_grad.\n\n\n\n"
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
    "location": "reference/#GpAbc.scaled_squared_distance-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpAbc.scaled_squared_distance",
    "category": "method",
    "text": "scaled_squared_distance(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nCompute the scaled squared distance between x and z:\n\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\n\nThe gradient of this function with respect to length scale hyperparameter(s) is returned by scaled_squared_distance_grad.\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\n\nReturn\n\nAn n times m matrix of scaled squared distances\n\n\n\n"
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
    "location": "reference/#GpAbc-1",
    "page": "Reference",
    "title": "GpAbc",
    "category": "section",
    "text": "Modules = [GpAbc]\nOrder   = [:type, :function, :method]"
},

]}
