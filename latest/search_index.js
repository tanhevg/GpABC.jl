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
    "text": "using GpABC, Distributions, PyPlot\n\n# prepare the data\nn = 30\nf(x) = x.ˆ2 + 10 * sin.(x) # the latent function\n\ntraining_x = sort(rand(Uniform(-10, 10), n))\ntraining_y = f(training_x)\ntraining_y += 20 * (rand(n) - 0.5) # add some noise\ntest_x = collect(linspace(min(training_x...), max(training_x...), 1000))\n\n # SquaredExponentialIsoKernel is used by default\ngpm = GPModel(training_x, training_y)\n\n# pretend we know the hyperparameters in advance\n# σ_f = 37.08; l = 1.0; σ_n = 6.58. See SquaredExponentialIsoKernel documentation for details\nset_hyperparameters(gpm, [37.08, 1.0, 6.58])\n(test_y, test_var) = gp_regression(test_x, gpm)\n\nplot(test_x, [test_y f(test)]) # ... and more sophisticated plotting"
},

{
    "location": "examples/#example-2-1",
    "page": "Examples",
    "title": "Optimising Hyperparameters for GP Regression",
    "category": "section",
    "text": "Based on Basic Gaussian Process Regression, but with added optimisation of hyperparameters:using GpABC\n\n# prepare the data ...\n\ngpm = GPModel(training_x, training_y)\n\n # by default, the optimiser will start with all hyperparameters set to 1,\n # constrained between exp(-10) and exp(10)\ntheta_mle = gp_train(gpm)\n\n# optimised hyperparameters are stored in gpm, so no need to pass them again\n(test_y, test_var) = gp_regression(test_x, gpm)"
},

{
    "location": "examples/#example-3-1",
    "page": "Examples",
    "title": "Advanced Usage of gp_train",
    "category": "section",
    "text": "using GpABC, Optim, Distributions\n\nfunction gp_train_advanced(gpm::GPModel, attempts::Int)\n    # Initialise the bounds, with special treatment for the second hyperparameter\n    p = get_hyperparameters_size(gpm)\n    bound_template = ones(p)\n    upper_bound = bound_template * 10\n    upper_bound[2] = 2\n    lower_bound = bound_template * -10\n    lower_bound[2] = -1\n\n    # Starting point will be sampled from a Multivariate Uniform distribution\n    start_point_distr = MvUniform(lower_bound, upper_bound)\n\n    # Run several attempts of training and store the\n    # minimiser hyperparameters and the value of log likelihood function\n    hypers = Array{Float64}(attempts, p)\n    likelihood_values = Array{Float64}(attempts)\n    for i=1:attempts\n        set_hyperparameters(gpm, exp.(rand(start_point_distr)))\n        hypers[i, :] = gp_train(gpm,\n            optimisation_solver_type=SimulatedAnnealing, # note the solver type\n            hp_lower=lower_bound, hp_upper=upper_bound, log_level=1)\n        likelihood_values[i] = gp_loglikelihood(gpm)\n    end\n    # Retain the hyperparameters where the maximum log likelihood function is attained\n    gpm.gp_hyperparameters = hypers[indmax(likelihood_values), :]\nend"
},

{
    "location": "examples/#example-4-1",
    "page": "Examples",
    "title": "Using a Custom Kernel",
    "category": "section",
    "text": "The methods below should be implemented for the custom kernel, unless indicated as optional. Please see reference documentation for detailed description of each method and parameter.using GpABC\nimport GpABC.covariance, GpABC.get_hyperparameters_size, GpABC.covariance_diagonal,\n    GpABC.covariance_training, GpABC.covariance_grad\n\n\"\"\"\n   This is the new kernel that we are adding\n\"\"\"\ntype MyCustomkernel <: AbstractGPKernel\n\n    # optional cache of matrices that could be re-used between calls to\n    # covariance_training and covariance_grad, keyed by hyperparameters\n    cache::MyCustomCache\nend\n\n\"\"\"\n    Report the number of hyperparameters required by the new kernel\n\"\"\"\nfunction get_hyperparameters_size(ker::MyCustomkernel, training_data::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n    Covariance function of the new kernel.\n\n    Return the covariance matrix. Assuming x is an n by d matrix, and z is an m by d matrix,\n    this should return an n by m matrix. Use `scaled_squared_distance` helper function here.\n\"\"\"\nfunction covariance(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n    Optional speedup of `covariance` function, that is invoked when the calling code is\n    only interested in variance (i.e. diagonal elements of the covariance) of the kernel.\n\"\"\"\nfunction covariance_diagonal(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n   Optional speedup of `covariance` function that is invoked during training of the GP.\n   Intermediate matrices that are re-used between this function and `covariance_grad` could\n   be cached in `ker.cache`\n\"\"\"\nfunction covariance_training(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    training_x::AbstractArray{Float64, 2})\n    # ...\nend\n\n\"\"\"\n    Optional gradient of `covariance` function with respect to hyperparameters, required\n    for optimising with `ConjugateGradient` method. If not provided, `NelderMead` optimiser\n    will be used.\n\n    Use `scaled_squared_distance_grad` helper function here.\n\"\"\"\nfunction covariance_grad(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n    # ...\nend\n\ngpm = GPModel(training_x, training_y, MyCustomkernel())\ntheta = gp_train(gpm)\n(test_y, test_var) = gp_regression(test_x, gpm)"
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
    "text": "Modules = [GpABC]"
},

{
    "location": "reference/#GpABC.ABCRejectionOutput",
    "page": "Reference",
    "title": "GpABC.ABCRejectionOutput",
    "category": "type",
    "text": "ABCRejectionOutput\n\nA container for the output of a rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_accepted::Int64: The number of accepted parameter vectors (particles) in the posterior.\nn_tries::Int64: The total number of parameter vectors (particles) that were tried.\nthreshold::Float64: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\npopulation::AbstractArray{Float64,2}: The parameter vectors (particles) in the posterior. Size: (n_accepted, n_params).\ndistances::AbstractArray{Float64,1}: The distances for each parameter vector (particle) in the posterior to the observed data in summary statistic space. Size: (n_accepted).\nweights::StatsBase.Weights: The weight of each parameter vector (particle) in the posterior.\n\n\n\n"
},

{
    "location": "reference/#GpABC.ABCSMCOutput",
    "page": "Reference",
    "title": "GpABC.ABCSMCOutput",
    "category": "type",
    "text": "ABCSMCOutput\n\nA container for the output of a rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_accepted::Int64: The number of accepted parameter vectors (particles) in the posterior.\nn_tries::Int64: The total number of parameter vectors (particles) that were tried.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\npopulation::AbstractArray{Float64,2}: The parameter vectors (particles) in the posterior. Size: (n_accepted, n_params).\ndistances::AbstractArray{Float64,1}: The distances for each parameter vector (particle) in the posterior to the observed data in summary statistic space. Size: (n_accepted).\nweights::StatsBase.Weights: The weight of each parameter vector (particle) in the posterior.\n\n\n\n"
},

{
    "location": "reference/#GpABC.AbstractGPKernel",
    "page": "Reference",
    "title": "GpABC.AbstractGPKernel",
    "category": "type",
    "text": "AbstractGPKernel\n\nAbstract kernel type. User-defined kernels should derive from it.\n\nImplementations have to provide methods for get_hyperparameters_size and covariance. Methods for covariance_training, covariance_diagonal and covariance_grad are optional.\n\n\n\n"
},

{
    "location": "reference/#GpABC.EmulatedABCRejectionInput",
    "page": "Reference",
    "title": "GpABC.EmulatedABCRejectionInput",
    "category": "type",
    "text": "EmulatedABCRejectionInput\n\nAn object that defines the settings for a emulation-based rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold::Float64: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: A 1D Array of distributions with length n_params from which candidate parameter vectors will be sampled.\nbatch_size::Int64: The number of predictions to be made in each batch.\nmax_iter::Int64: The maximum number of iterations/batches before termination.\n\n\n\n"
},

{
    "location": "reference/#GpABC.EmulatedABCSMCInput",
    "page": "Reference",
    "title": "GpABC.EmulatedABCSMCInput",
    "category": "type",
    "text": "EmulatedABCRejectionInput\n\nAn object that defines the settings for a emulation-based rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated (the length of each parameter vector/particle).\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: A 1D Array of distributions with length n_params from which candidate parameter vectors will be sampled.\nbatch_size::Int64: The number of predictions to be made in each batch.\nmax_iter::Int64: The maximum number of iterations/batches before termination.\n\n\n\n"
},

{
    "location": "reference/#GpABC.GPModel",
    "page": "Reference",
    "title": "GpABC.GPModel",
    "category": "type",
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        kernel::AbstractGPKernel\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nConstructor of GPModel that allows the kernel to be specified. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
},

{
    "location": "reference/#GpABC.GPModel",
    "page": "Reference",
    "title": "GpABC.GPModel",
    "category": "type",
    "text": "GPModel\n\nThe main type that is used by most functions within the package.\n\nAll data matrices are row-major.\n\nFields\n\nkernel::AbstractGPKernel: the kernel\ngp_training_x::AbstractArray{Float64, 2}: training x. Size: n times d.\ngp_training_y::AbstractArray{Float64, 2}: training y. Size: n times 1.\ngp_test_x::AbstractArray{Float64, 2}: test x.  Size: m times d.\ngp_hyperparameters::AbstractArray{Float64, 1}: kernel hyperparameters, followed by standard deviation of intrinsic noise sigma_n, which is always the last element in the array.\ncache::HPOptimisationCache: cache of matrices that can be re-used between calls to gp_loglikelihood and gp_loglikelihood_grad\n\n\n\n"
},

{
    "location": "reference/#GpABC.GPModel",
    "page": "Reference",
    "title": "GpABC.GPModel",
    "category": "type",
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nDefault constructor of GPModel, that will use SquaredExponentialIsoKernel. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
},

{
    "location": "reference/#GpABC.GPModel-Tuple{}",
    "page": "Reference",
    "title": "GpABC.GPModel",
    "category": "method",
    "text": "GPModel(;training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    kernel::AbstractGPKernel=SquaredExponentialIsoKernel(),\n    gp_hyperparameters::AbstractArray{Float64, 1}=Array{Float64}(0))\n\nConstructor of GPModel with explicit arguments. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n"
},

{
    "location": "reference/#GpABC.MaternArdKernel",
    "page": "Reference",
    "title": "GpABC.MaternArdKernel",
    "category": "type",
    "text": "MaternArdKernel <: AbstractGPKernel\n\nMatérn kernel with distinct length scale for each dimention, l_k. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n"
},

{
    "location": "reference/#GpABC.MaternIsoKernel",
    "page": "Reference",
    "title": "GpABC.MaternIsoKernel",
    "category": "type",
    "text": "MaternIsoKernel <: AbstractGPKernel\n\nMatérn kernel with uniform length scale across all dimensions, l. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n"
},

{
    "location": "reference/#GpABC.ModelSelectionOutput",
    "page": "Reference",
    "title": "GpABC.ModelSelectionOutput",
    "category": "type",
    "text": "ModelSelectionOutput\n\nContains results of a model selection computation, including which models are best supported by the observed data and the parameter poseteriors at each population for each model.\n\nFields\n\nM::Int64: The number of models.\nn_accepted::AbstractArray{AbstractArray{Int64,1},1}: The number of parameters accepted by each model at each population. n_accepted[i][j] contains the number of acceptances for model j at population i.\nthreshold_schedule::AbstractArray{Float64,1}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\nsmc_outputs::AbstractArray{ABCSMCOutput,1}: A [\'SimulatedABCSMCOutput\']@(ref) or [\'EmulatedABCSMCOutput\']@(ref) for each model. Use to find details of the ABC results at each population.\n\n\n\n"
},

{
    "location": "reference/#GpABC.RepetitiveTraining",
    "page": "Reference",
    "title": "GpABC.RepetitiveTraining",
    "category": "type",
    "text": "RepetitiveTraining\n\nA structure that holds the settings for repetitive training of the emulator. On each iteration of re-training a certain number of points (rt_sample_size) is sampled from the prior. Variance of emulator prediction is then obtained for this sample. Particles with the highest variance are added to the training set. The model is then simulated for this additional parameters, and the emulator is re-trained.\n\nFields\n\nrt_iterations: Number of times the emulator will be re-trained.\nrt_sample_size: Size of the sample that will be used to evaluate emulator variance.\nrt_extra_training_points: Number of particles with highest variance to add to the training set on each iteration.\n\n\n\n"
},

{
    "location": "reference/#GpABC.SimulatedABCRejectionInput",
    "page": "Reference",
    "title": "GpABC.SimulatedABCRejectionInput",
    "category": "type",
    "text": "SimulatedABCRejectionInput\n\nAn object that defines the settings for a simulation-based rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold::Float64: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: A 1D Array of distributions with length n_params from which candidate parameter vectors will be sampled.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays.\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\n\n\n\n"
},

{
    "location": "reference/#GpABC.SimulatedABCSMCInput",
    "page": "Reference",
    "title": "GpABC.SimulatedABCSMCInput",
    "category": "type",
    "text": "SimulatedABCSMCInput\n\nAn object that defines the settings for a simulation-based ABC-SMC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: A 1D Array of distributions with length n_params from which candidate parameter vectors will be sampled.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays.\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\nmax_iter::Integer: The maximum number of iterations in each population before algorithm termination.\n\n\n\n"
},

{
    "location": "reference/#GpABC.SimulatedModelSelectionInput",
    "page": "Reference",
    "title": "GpABC.SimulatedModelSelectionInput",
    "category": "type",
    "text": "SimulatedModelSelectionInput\n\nAn object that defines settings for a simulation-based model selection computation.\n\nFields\n\nM::Int64: The number of models.\nn_particles::Int64: The number of particles to be accepted per population (at the model level)\nthreshold_schedule::AbstractArray{Float64,1}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\nmodel_prior::DiscreteUnivariateDistribution: The prior from which models will be sampled.\nparameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution,1},1}: Parameter priors for each model. Each element is an array of priors for the corresponding model (one prior per parameter).\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays.\nsimulator_functions::AbstractArray{Function,1}: Each element is a function that takes a parameter vector as an argument and outputs model results for a single model.\nmax_iter::Integer: The maximum number of iterations in each population before algorithm termination.\n\n\n\n"
},

{
    "location": "reference/#GpABC.SquaredExponentialArdKernel",
    "page": "Reference",
    "title": "GpABC.SquaredExponentialArdKernel",
    "category": "type",
    "text": "SquaredExponentialArdKernel <: AbstractGPKernel\n\nSquared exponential kernel with distinct length scale for each dimention, l_k.\n\nbeginaligned\nK(r)  = sigma_f^2 e^-r2 \nr_ij  = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n"
},

{
    "location": "reference/#GpABC.SquaredExponentialIsoKernel",
    "page": "Reference",
    "title": "GpABC.SquaredExponentialIsoKernel",
    "category": "type",
    "text": "SquaredExponentialIsoKernel <: AbstractGPKernel\n\nSquared exponential kernel with uniform length scale across all dimensions, l.\n\nbeginaligned\nK(r)  = sigma_f^2 e^-r2 \nr_ij  = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n"
},

{
    "location": "reference/#GpABC.ABCSMC-Tuple{GpABC.ABCSMCInput,AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.ABCSMC",
    "category": "method",
    "text": "ABCSMC\n\nRun a ABC-SMC computation using either simulation (the model is simulated in full for each parameter vector from which the corresponding distance to observed data is used to construct the posterior) or emulation (a regression model trained to predict the distance from the parameter vector directly is used to construct the posterior). Whether simulation or emulation is used is controlled by the type of input.\n\nArguments\n\ninput::ABCSMCInput: An \'SimulatedABCSMCInput\' or \'EmulatedABCSMCInput\' object that defines the settings for the ABC-SMC run.\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nwrite_progress::Bool: Optional argument controlling whether progress is written to out_stream.\nprogress_every::Int: Progress will be written to out_stream every progress_every simulations (optional, ignored if write_progress is False).\n\nReturn\n\nAn object that inherits from \'ABCSMCOutput\', depending on whether a input is a \'SimulatedABCSMCInput\' or \'EmulatedABCSMCInput\'.\n\n\n\n"
},

{
    "location": "reference/#GpABC.ABCrejection-Tuple{GpABC.EmulatedABCRejectionInput,AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.ABCrejection",
    "category": "method",
    "text": "ABCrejection\n\nRun a emulation-based rejection-ABC computation. Parameter posteriors are obtained using a regression model (the emulator), that has learnt a mapping from parameter vectors to the distance between the model output and observed data in summary statistic space. If this distance is sufficiently small the parameter vector is included in the posterior.\n\nArguments\n\ninput::EmulatedABCRejectionInput: An \'EmulatedABCRejectionInput\' object that defines the settings for the emulated rejection-ABC run.\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nwrite_progress::Bool: Optional argument controlling whether progress is logged.\nprogress_every::Int: Progress will be logged every progress_every simulations (optional, ignored if write_progress is False).\n\nReturns\n\nAn \'EmulatedABCRejectionOutput\' object.\n\n\n\n"
},

{
    "location": "reference/#GpABC.ABCrejection-Tuple{GpABC.SimulatedABCRejectionInput,AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.ABCrejection",
    "category": "method",
    "text": "ABCrejection\n\nRun a simulationed-based rejection-ABC computation. Parameter posteriors are obtained by simulating the model for a parameter vector, computing the summary statistic of the output then computing the distance to the summary statistic of the reference data. If this distance is sufficiently small the parameter vector is included in the posterior.\n\nArguments\n\ninput::SimulatedABCRejectionInput: A \'SimulatedABCRejectionInput\' object that defines the settings for the simulated rejection-ABC run.\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nwrite_progress::Bool: Optional argument controlling whether progress is written to out_stream.\nprogress_every::Int: Progress will be written to out_stream every progress_every simulations (optional, ignored if write_progress is False).\n\nReturns\n\nA \'SimulatedABCRejectionOutput\' object.\n\n\n\n"
},

{
    "location": "reference/#GpABC.EmulatedABCRejection-Union{Tuple{D}, Tuple{Int64,AbstractArray{Float64,2},Int64,Float64,AbstractArray{D,1},Union{AbstractArray{String,1}, Function, String},Function}} where D<:Distributions.Distribution{Distributions.Univariate,Distributions.Continuous}",
    "page": "Reference",
    "title": "GpABC.EmulatedABCRejection",
    "category": "method",
    "text": "EmulatedABCRejection\n\nA convenience function that trains a Gaussian process emulator of  type `GPmodel then uses it in emulation-based rejection-ABC. It creates the training data by simulating the model for the design points, trains the emulator, creates the EmulatedABCRejectionInput object then calls `ABCrejection.\n\nFields\n\nn_design_points::Int64: The number of parameter vectors used to train the Gaussian process emulator\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold::Float64: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\npriors::AbstractArray{D,1}: A 1D Array of continuous univariate distributions with length n_params from which candidate parameter vectors will be sampled.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\ndistance_metric::Function: Any function that computes the distance between 2 1D Arrays (optional - default is to use the Euclidean distance).\ngpkernel::AbstractGPKernel: An object inheriting from AbstractGPKernel that is the Gaussian process kernel. (optional - default is the ARD-RBF/squared exponential kernel).\nbatch_size::Int64: The number of predictions to be made in each batch (optional - default is 10 times n_particles).\nmax_iter::Int64: The maximum number of iterations/batches before termination.\nkwargs: optional keyword arguments passed to \'ABCrejection\'.\n\n\n\n"
},

{
    "location": "reference/#GpABC.EmulatedABCSMC-Union{Tuple{D}, Tuple{Int64,AbstractArray{Float64,2},Int64,AbstractArray{Float64,1},AbstractArray{D,1},Union{AbstractArray{String,1}, Function, String},Function}} where D<:Distributions.Distribution{Distributions.Univariate,Distributions.Continuous}",
    "page": "Reference",
    "title": "GpABC.EmulatedABCSMC",
    "category": "method",
    "text": "EmulatedABCSMC\n\nA convenience function that trains a Gaussian process emulator of type `GPmodel then uses it in emulation-based ABC-SMC. It creates the training data by simulating the model for the design points, trains the emulator, creates the EmulatedABCSMCInput object then calls ABCSMC.\n\nFields\n\nn_design_points::Int64: The number of parameter vectors used to train the Gaussian process emulator\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\npriors::AbstractArray{D,1}: A 1D Array of continuous univariate distributions with length n_params from which candidate parameter vectors will be sampled.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarise model output. REFER TO DOCS\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\ndistance_metric::Function: Any function that computes the distance between 2 1D Arrays (optional - default is to use the Euclidean distance).\ngpkernel::AbstractGPKernel: An object inheriting from AbstractGPKernel that is the Gaussian process kernel. (optional - default is the ARD-RBF/squared exponential kernel).\nbatch_size::Int64: The number of predictions to be made in each batch (optional - default is 10 times n_particles).\nmax_iter::Int64: The maximum number of iterations/batches before termination.\nkwargs: optional keyword arguments passed to \'ABCSMC\'.\n\n\n\n"
},

{
    "location": "reference/#GpABC.ExponentialArdKernel-Tuple{}",
    "page": "Reference",
    "title": "GpABC.ExponentialArdKernel",
    "category": "method",
    "text": "ExponentialArdKernel\n\nAlias for MaternArdKernel(1)\n\n\n\n"
},

{
    "location": "reference/#GpABC.ExponentialIsoKernel-Tuple{}",
    "page": "Reference",
    "title": "GpABC.ExponentialIsoKernel",
    "category": "method",
    "text": "ExponentialIsoKernel\n\nAlias for MaternIsoKernel(1)\n\n\n\n"
},

{
    "location": "reference/#GpABC.SimulatedABCRejection-Union{Tuple{AbstractArray{Float64,2},Int64,Float64,AbstractArray{D,1},Union{AbstractArray{String,1}, Function, String},Function}, Tuple{D}} where D<:Distributions.Distribution{Distributions.Univariate,Distributions.Continuous}",
    "page": "Reference",
    "title": "GpABC.SimulatedABCRejection",
    "category": "method",
    "text": "SimulatedABCRejection\n\nRun a simulation-based ABC-rejection computation. This is a convenience wrapper that constructs a SimulatedABCRejectionInput object then calls \'ABCrejection\'.\n\nArguments\n\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\npriors::AbstractArray{D,1}: A 1D Array of continuous univariate distributions with length n_params from which candidate parameter vectors will be sampled.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).\nmax_iter::Int64: The maximum number of simulations that will be run. The default is 1000*n_particles.\nkwargs: optional keyword arguments passed to \'ABCrejection\'.\n\nReturns\n\nA \'SimulatedABCRejectionOutput\' object.\n\n\n\n"
},

{
    "location": "reference/#GpABC.SimulatedABCSMC-Union{Tuple{AbstractArray{Float64,2},Integer,AbstractArray{Float64,1},AbstractArray{D,1},Union{AbstractArray{String,1}, Function, String},Function}, Tuple{D}} where D<:Distributions.Distribution{Distributions.Univariate,Distributions.Continuous}",
    "page": "Reference",
    "title": "GpABC.SimulatedABCSMC",
    "category": "method",
    "text": "SimulatedABCSMC\n\nRun a emulation-based ABC-rejection computation. This is a convenience wrapper that constructs a SimulatedABCSMCInput object then calls \'ABCrejection\'.\n\nArguments\n\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: A 1D Array of continuous univariate distributions with length n_params from which candidate parameter vectors will be sampled.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).\nmax_iter::Integer: The maximum number of simulations that will be run. The default is 1000*n_particles.\nkwargs: optional keyword arguments passed to \'ABCrejection\'.\n\nReturns\n\nA \'SimulatedABCSMCOutput\' object that contains the posteriors at each ABC-SMC population and other information.\n\n\n\n"
},

{
    "location": "reference/#GpABC.covariance-Tuple{GpABC.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.covariance",
    "category": "method",
    "text": "covariance(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nReturn the covariance matrix. Should be overridden by kernel implementations.\n\nArguments\n\nker: The kernel object. Implementations must override with their own subtype.\nlog_theta: natural logarithm of hyperparameters.\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\n\nReturn\n\nThe covariance matrix, of size n times m.\n\n\n\n"
},

{
    "location": "reference/#GpABC.covariance_diagonal-Tuple{GpABC.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.covariance_diagonal",
    "category": "method",
    "text": "covariance_diagonal(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2})\n\nThis is a speedup version of covariance, which is invoked if the caller is not interested in the entire covariance matrix, but only needs the variance, i.e. the diagonal of the covariance matrix.\n\nDefault method just returns diag(covariance(...)), with x === z. Kernel implementations can optionally override it to achieve betrer performance, by not computing the non diagonal elements of covariance matrix.\n\nSee covariance for description of arguments.\n\nReturn\n\nThe 1-d array of variances, of size size(x, 1).\n\n\n\n"
},

{
    "location": "reference/#GpABC.covariance_grad-Tuple{GpABC.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.covariance_grad",
    "category": "method",
    "text": "covariance_grad(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n\nReturn the gradient of the covariance function with respect to logarigthms of hyperparameters, based on the provided direction matrix.\n\nThis function can be optionally overridden by kernel implementations. If the gradient function is not provided, gp_train will fail back to NelderMead algorithm by default.\n\nArguments\n\nker: The kernel object. Implementations must override with their own subtype.\nlog_theta:  natural logarithm of hyperparameters\nx: Training data, reshaped into a 2-d array. x must have dimensions n times d.\nR the directional matrix, n times n\n\nR = frac1sigma_n^2(alpha * alpha^T - K^-1) alpha = K^-1y\n\nReturn\n\nA vector of size length(log_theta), whose j\'th element is equal to\n\ntr(R fracpartial Kpartial eta_j)\n\n\n\n"
},

{
    "location": "reference/#GpABC.covariance_training-Tuple{GpABC.AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.covariance_training",
    "category": "method",
    "text": "covariance_training(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    training_x::AbstractArray{Float64, 2})\n\nThis is a speedup version of covariance, which is only called during traing sequence. Intermediate matrices computed in this function for particular hyperparameters can be cached and reused subsequently, either in this function or in covariance_grad\n\nDefault method just delegates to covariance with x === z. Kernel implementations can optionally override it for betrer performance.\n\nSee covariance for description of arguments and return values.\n\n\n\n"
},

{
    "location": "reference/#GpABC.get_hyperparameters_size-Tuple{GpABC.AbstractGPKernel,AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.get_hyperparameters_size",
    "category": "method",
    "text": "get_hyperparameters_size(kernel::AbstractGPKernel, training_data::AbstractArray{Float64, 2})\n\nReturn the number of hyperparameters for used by this kernel on this training data set. Should be overridden by kernel implementations.\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_loglikelihood-Tuple{AbstractArray{Float64,1},GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_loglikelihood",
    "category": "method",
    "text": "gp_loglikelihood(theta::AbstractArray{Float64, 1}, gpm::GPModel)\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_loglikelihood-Tuple{GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_loglikelihood",
    "category": "method",
    "text": "gp_loglikelihood(gpm::GPModel)\n\nCompute the log likelihood function, based on the kernel and training data specified in gpm.\n\nlog p(y vert X theta) = - frac12(y^TK^-1y + log vert K vert + n log 2 pi)\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_loglikelihood_grad-Tuple{AbstractArray{Float64,1},GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_loglikelihood_grad",
    "category": "method",
    "text": "gp_loglikelihood_grad(theta::AbstractArray{Float64, 1}, gpem::GPModel)\n\nGradient of the log likelihood function (gp_loglikelihood_log) with respect to logged hyperparameters.\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_loglikelihood_log-Tuple{AbstractArray{Float64,1},GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_loglikelihood_log",
    "category": "method",
    "text": "gp_loglikelihood_log(theta::AbstractArray{Float64, 1}, gpm::GPModel)\n\nLog likelihood function with log hyperparameters. This is the target function of the hyperparameters optimisation procedure. Its gradient is coputed by gp_loglikelihood_grad.\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_regression-Tuple{GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_regression",
    "category": "method",
    "text": "gp_regression(gpm::GPModel; <optional keyword arguments>)\n\nRun the Gaussian Process Regression.\n\nArguments\n\ngpm: the GPModel, that contains the training data (x and y), the kernel, the hyperparameters and the test data for running the regression.\ntest_x: if specified, overrides the test x in gpm. Size m times d.\nlog_level::Int (optional): log level. Default is 0, which is no logging at all. 1 makes gp_regression print basic information to standard output.\nfull_covariance_matrix::Bool (optional): whether we need the full covariance matrix, or just the variance vector. Defaults to false (i.e. just the variance).\nbatch_size::Int (optional): If full_covariance_matrix is set to false, then the mean and variance vectors will be computed in batches of this size, to avoid allocating huge matrices. Defaults to 1000.\nobservation_noise::Bool (optional): whether the observation noise (with variance sigma_n^2) should be included in the output variance. Defaults to true.\n\nReturn\n\nA tuple of (mean, var). mean is a mean vector of the output multivariate Normal distribution, of size m. var is either the covariance matrix of size m times m, or a variance vector of size m, depending on full_covariance_matrix flag.\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_regression-Tuple{Union{AbstractArray{Float64,1}, AbstractArray{Float64,2}},GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_regression",
    "category": "method",
    "text": "gp_regression(test_x::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}},\n    gpem::GPModel; <optional keyword arguments>)\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_regression_sample-Tuple{Union{AbstractArray{Float64,1}, AbstractArray{Float64,2}},GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_regression_sample",
    "category": "method",
    "text": "gp_regression_sample(params::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}}, gpem::GPModel)\n\nReturn a random sample from a multivariate Gaussian distrubution, obtained by calling gp_regression\n\n\n\n"
},

{
    "location": "reference/#GpABC.gp_train-Tuple{GpABC.GPModel}",
    "page": "Reference",
    "title": "GpABC.gp_train",
    "category": "method",
    "text": "gp_train(gpm::GPModel; <optional keyword arguments>)\n\nFind Maximum Likelihood Estimate of Gaussian Process hyperparameters by maximising gp_loglikelihood, using Optim package.\n\nArguments\n\ngpm: the GPModel, that contains the training data (x and y), the kernel and the starting hyperparameters that will be used for optimisation.\noptimisation_solver_type::Type{<:Optim.Optimizer} (optional): the solver to use. If not given, then ConjugateGradient will be used for kernels that have gradient implementation, and NelderMead will be used for those that don\'t.\nhp_lower::AbstractArray{Float64, 1} (optional): the lower boundary for box optimisation. Defaults to e^-10 for all hyperparameters.\nhp_upper::AbstractArray{Float64, 1} (optional): the upper boundary for box optimisation. Defaults to e^10 for all hyperparameters.\nlog_level::Int (optional): log level. Default is 0, which is no logging at all. 1 makes gp_train print basic information to standard output. 2 switches Optim logging on, in addition to 1.\n\nReturn\n\nThe list of all hyperparameters, including the standard deviation of the measurement noise sigma_n. Note that after this function returns, the hyperparameters of gpm will be set to the optimised value, and there is no need to call set_hyperparameters once again.\n\n\n\n"
},

{
    "location": "reference/#GpABC.model_selection-Tuple{GpABC.SimulatedModelSelectionInput,AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.model_selection",
    "category": "method",
    "text": "model_selection(input::SimulatedModelSelectionInput,\n	reference_data::AbstractArray{Float64,2})\n\nArguments\n\ninput::SimulatedModelSelectionInput: A [\'SimulatedModelSelectionInput\']@(ref) object that contains the settings for the model selection algorithm.\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\n\n\n\n"
},

{
    "location": "reference/#GpABC.model_selection-Union{Tuple{AD}, Tuple{AbstractArray{Float64,2},Integer,AbstractArray{Float64,1},AbstractArray{AD,1},Union{AbstractArray{String,1}, Function, String},AbstractArray{Function,1}}, Tuple{D}} where AD<:AbstractArray{D,1} where D<:Distributions.Distribution{Distributions.Univariate,Distributions.Continuous}",
    "page": "Reference",
    "title": "GpABC.model_selection",
    "category": "method",
    "text": "model_selection\n\nPerform model selection using simulation-based ABC.\n\nArguments\n\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\nparameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution},1}: Priors for the parameters of each model. The length of the outer array is the number of models.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\nsimulator_functions::AbstractArray{Function,1}: An array of functions that take a parameter vector as an argument and outputs model results (one per model).\n\'model_prior::DiscreteUnivariateDistribution\': The prior from which models are sampled. Default is a discrete, uniform distribution.\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).\nmax_iter::Integer: The maximum number of simulations that will be run. The default is 1000*n_particles. Each iteration samples a single model and performs ABC using a single particle.\n\nReturns\n\nA \'ModelSelectionOutput\' object that contains which models are supported by the observed data.\n\n\n\n"
},

{
    "location": "reference/#GpABC.model_selection-Union{Tuple{AD}, Tuple{D}, Tuple{Int64,AbstractArray{Float64,2},Int64,AbstractArray{Float64,1},AbstractArray{AD,1},Union{AbstractArray{String,1}, Function, String},AbstractArray{Function,1}}} where AD<:AbstractArray{D,1} where D<:Distributions.Distribution{Distributions.Univariate,Distributions.Continuous}",
    "page": "Reference",
    "title": "GpABC.model_selection",
    "category": "method",
    "text": "model_selection\n\nPerform model selection using emulation-based ABC.\n\nArguments\n\nn_design_points::Int64: The number of parameter vectors used to train the Gaussian process emulator.\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (n_model_trajectories, n_time_points)\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\nparameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution},1}: Priors for the parameters of each model. The length of the outer array is the number of models.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS\nsimulator_functions::AbstractArray{Function,1}: An array of functions that take a parameter vector as an argument and outputs model results (one per model).\n\'model_prior::DiscreteUnivariateDistribution\': The prior from which models are sampled. Default is a discrete, uniform distribution.\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).\nmax_iter::Integer: The maximum number of simulations that will be run. The default is 1000*n_particles. Each iteration samples a single model and performs ABC using a single particle.\nmax_batch_size::Integer: The maximum batch size for the emulator when making predictions.\n\nReturns\n\nA \'ModelSelectionOutput\' object that contains which models are supported by the observed data.\n\n\n\n"
},

{
    "location": "reference/#GpABC.scaled_squared_distance-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.scaled_squared_distance",
    "category": "method",
    "text": "scaled_squared_distance(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nCompute the scaled squared distance between x and z:\n\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\n\nThe gradient of this function with respect to length scale hyperparameter(s) is returned by scaled_squared_distance_grad.\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\n\nReturn\n\nAn n times m matrix of scaled squared distances\n\n\n\n"
},

{
    "location": "reference/#GpABC.scaled_squared_distance_grad-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Reference",
    "title": "GpABC.scaled_squared_distance_grad",
    "category": "method",
    "text": "scaled_squared_distance_grad(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n\nReturn the gradient of the scaled_squared_distance function with respect to logarigthms of length scales, based on the provided direction matrix.\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\nR the direction matrix, n times m. This can be used to compute the gradient of a function that depends on scaled_squared_distance via the chain rule.\n\nReturn\n\nA vector of size length(log_ell), whose k\'th element is equal to\n\ntexttr(R fracpartial Kpartial l_k)\n\n\n\n"
},

{
    "location": "reference/#GpABC.set_hyperparameters-Tuple{GpABC.GPModel,AbstractArray{Float64,1}}",
    "page": "Reference",
    "title": "GpABC.set_hyperparameters",
    "category": "method",
    "text": "set_hyperparameters(gpm::GPModel, hypers::AbstractArray{Float64, 1})\n\nSet the hyperparameters of the GPModel\n\n\n\n"
},

{
    "location": "reference/#GpABC-1",
    "page": "Reference",
    "title": "GpABC",
    "category": "section",
    "text": "Modules = [GpABC]\nOrder   = [:type, :function, :method]"
},

]}
