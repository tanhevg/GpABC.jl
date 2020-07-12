var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#GpABC-1",
    "page": "Home",
    "title": "GpABC",
    "category": "section",
    "text": "GpABC provides algorithms for likelihood - free parameter inference and model selection using Approximate Bayesian Computation (ABC). Two sets of algorithms are available:Simulation based - full simulations of the model(s) is done on each step of ABC.\nEmulation based - a small number of simulations can be used to train a regression model (the emulator), which is then used to approximate model simulation results during ABC.GpABC offers Gaussian Process Regression (GPR) as an emulator, but custom emulators can also be used. GPR can also be used standalone, for any regression task.Stochastic models, that don\'t conform to Gaussian Process Prior assumption, are supported via Linear Noise Approximation (LNA)."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "GpABC can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and runpkg> add https://github.com/tanhevg/GpABC.jl"
},

{
    "location": "#Notation-1",
    "page": "Home",
    "title": "Notation",
    "category": "section",
    "text": "In parts of this manual that deal with Gaussian Processes and kernels, we denote the number of training points as n, and the number of test points as m. The number of dimensions is denoted as d.In the context of ABC, vectors in parameter space (theta) are referred to as particles. Particles that are used for training the emulator (training_x) are called design points. To generate the distances for training the emulator (training_y), the model must be simulated for the design points."
},

{
    "location": "#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": "ABC parameter estimation example\nABC model selection example\nStochastic Inference (LNA) example\nGaussian Process regression example"
},

{
    "location": "#Dependencies-1",
    "page": "Home",
    "title": "Dependencies",
    "category": "section",
    "text": "Optim - for training Gaussian Process hyperparameters.\nDistributions - probability distributions.\nDistances - distance functions\nOrdinaryDiffEq - for solving ODEs for LNA, and also used throughout the examples for model simulation (ODEs and SDEs)\nForwardDiff - automatic differentiation is also used by LNA"
},

{
    "location": "#References-1",
    "page": "Home",
    "title": "References",
    "category": "section",
    "text": "Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. Interface, (July 2008), 187–202. https://doi.org/10.1098/rsif.2008.0172\nFilippi, S., Barnes, C. P., Cornebise, J., & Stumpf, M. P. H. (2013). On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. Statistical Applications in Genetics and Molecular Biology, 12(1), 87–107. https://doi.org/10.1515/sagmb-2012-0069\nRasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. ISBN 0-262-18253-X. http://www.gaussianprocess.org/gpml\nSchnoerr, D., Sanguinetti, G., & Grima, R. (2017). Approximation and inference methods for stochastic biochemical kinetics—a tutorial review. Journal of Physics A: Mathematical and Theoretical, 50(9), 093001. https://doi.org/10.1088/1751-8121/aa54d9\nKarlebach, G., & Shamir, R. (2008). Modelling and analysis of gene regulatory networks. Nature Reviews Molecular Cell Biology, 9(10), 770–780. https://doi.org/10.1038/nrm2503"
},

{
    "location": "overview-abc/#",
    "page": "ABC Parameter Inference",
    "title": "ABC Parameter Inference",
    "category": "page",
    "text": ""
},

{
    "location": "overview-abc/#abc-overview-1",
    "page": "ABC Parameter Inference",
    "title": "ABC Overview",
    "category": "section",
    "text": "Approximate Bayesian Computation (ABC) is a collection of methods for likelihood free model parameter inference.Pages = [\"overview-abc.md\"]"
},

{
    "location": "overview-abc/#Simulation-based-Rejection-ABC-1",
    "page": "ABC Parameter Inference",
    "title": "Simulation based Rejection ABC",
    "category": "section",
    "text": "The most basic variant of ABC is referred to as Rejection ABC. The user-defined inputs to this algorithm include:The prior distribution pi, defined over model parameter space Theta\nThe model simulation function f\nReference data mathcalD\nAcceptance threshold varepsilon\nSummary statistic S and distance function d\nDesired size of the posterior sample and maximum number of simulations to performThe pseudocode for simulation-based Rejection ABC in GpABC looks as follows:While the posterior sample is not full, and maximum number of simulations has not been reached:\nSample parameter vector (particle) theta from pi\nSimulate data x = f(theta)\nCompute the distance between the summary statistic of the simulated data and that of the reference data y = d(S(x) S(mathcalD))\nIf y leq varepsilon, then accept theta in the posterior sampleThis algorithm is implemented by Julia function SimulatedABCRejection."
},

{
    "location": "overview-abc/#Emulation-based-Rejection-ABC-1",
    "page": "ABC Parameter Inference",
    "title": "Emulation based Rejection ABC",
    "category": "section",
    "text": "Some models are computationally expensive to simulate. Simulation based ABC for such models would take unreasonably long time to accept enough posterior particles.To work around this issue, GpABC provides emulation based Rejection ABC. Rather than simulating the model for each sampled particle, this algorithm runs a small number of simulations in the beginning, and uses their results to train the emulator.User-defined inputs for this algorithm are very similar to those for Simulation based Rejection ABC:The prior distribution pi, defined over model parameter space Theta\nThe model simulation function f\nReference data mathcalD\nAcceptance threshold varepsilon\nSummary statistic S and distance function d\nNumber of design particles to sample: n\nBatch size to use for regression: m\nDesired size of the posterior sample and maximum number of regressions to performThe pseudocode for emulation-based Rejection ABC in GpABC looks as follows:Sample n design particles from pi: theta_1 ldots theta_n\nSimulate the model for the design particles: x_1 ldots x_n = f(theta_1) ldots f(theta_n)\nCompute distances to the reference data: y_1 ldots y_n = d(S(x_1) S(mathcalD)) ldots d(S(x_n) S(mathcalD))\nUse theta_1 ldots theta_n and y_1 ldots y_n to train the emulator textbfgpr\nAdvanced: details of training procedure can be tweaked. See GpABC.train_emulator.\nWhile the posterior sample is not full, and maximum number of regressions has not been reached:\nSample m particles from pi: theta_1 ldots theta_m\nCompute the approximate distances by running the emulator regression: y_1 ldots y_m = textbfgpr(theta_1) ldots textbfgpr(theta_m)\nFor all j = 1 ldots m, if y_j leq varepsilon, then accept theta_j in the posterior sample\nAdvanced: details of the acceptance strategy can be tweaked. See GpABC.abc_select_emulated_particlesThis algorithm is implemented by Julia function EmulatedABCRejection."
},

{
    "location": "overview-abc/#Simulation-based-ABC-SMC-1",
    "page": "ABC Parameter Inference",
    "title": "Simulation based ABC - SMC",
    "category": "section",
    "text": "This sophisticated version of ABC allows to specify a schedule of thresholds, as opposed to just a single value. A number of simulation based ABC iterations are then executed, one iteration per threshold. The posterior of the preceding iteration serves as a prior to the next one.The user-defined inputs to this algorithm are similar to those of Simulation based Rejection ABC:The prior distribution pi, defined over model parameter space Theta\nThe model simulation function f\nReference data mathcalD\nA schedule of thresholds varepsilon_1 ldots varepsilon_T\nSummary statistic S and distance function d\nDesired size of the posterior sample and maximum number of simulations to performThe pseudocode for simulation-based ABC-SMC in GpABC looks as follows:For t in 1 ldots T\nWhile the posterior sample is not full, and maximum number of simulations has not been reached:\nif t = 1\nSample the particle theta from pi\nUse pdf of pi at each sampled particle as its weight w\nelse\nSample the particle theta from the posterior distribution of iteration t-1 with weights w\nPerturb theta using a perturbation kernel\nRecompute the weights w (see (Toni et al, 2009) for details)\nSimulate data x = f(theta)\nCompute the distance between the summary statistic of the simulated data and that of the reference data y = d(S(x) S(mathcalD))\nIf y leq varepsilon, then accept theta in the posterior sampleThis algorithm is implemented by Julia function SimulatedABCSMC."
},

{
    "location": "overview-abc/#Emulation-based-ABC-SMC-1",
    "page": "ABC Parameter Inference",
    "title": "Emulation based ABC - SMC",
    "category": "section",
    "text": "Similarly to Simulation based ABC - SMC, Emulation based Rejection ABC has an SMC counterpart. A threshold schedule must be supplied for this algorithm. A number of emulation based ABC iterations are then executed, one iteration per threshold. The posterior of the preceding iteration serves as a prior to the next one. Depending on user-defined settings, either the same emulator can be re-used for all iterations, or the emulator could be re-trained for each iteration.The user-defined inputs to this algorithm are similar to those of Emulation based Rejection ABC:The prior distribution pi, defined over model parameter space Theta\nThe model simulation function f\nReference data mathcalD\nA schedule of thresholds varepsilon_1 ldots varepsilon_T\nSummary statistic S and distance function d\nNumber of design particles to sample: n\nBatch size to use for regression: m\nDesired size of the posterior sample and maximum number of regressions to performThe pseudocode for emulation-based ABC-SMC in GpABC looks as follows:Sample n design particles from pi: theta_1 ldots theta_n\nSimulate the model for the design particles: x_1 ldots x_n = f(theta_1) ldots f(theta_n)\nCompute distances to the reference data: y_1 ldots y_n = d(S(x_1) S(mathcalD)) ldots d(S(x_n) S(mathcalD))\nUse theta_1 ldots theta_n and y_1 ldots y_n to train the emulator textbfgpr\nAdvanced: details of training procedure can be tweaked. See GpABC.train_emulator.\nFor t in 1 ldots T\nAdvanced: optionally, if t  1, re-traing the emulator. See GpABC.abc_retrain_emulator.\nWhile the posterior sample is not full, and maximum number of regressions has not been reached:\nif t = 1\nSample m particles from pi: theta_1 ldots theta_m\nUse pdf of pi at each sampled particle as its weight w\nelse\nSample m particles theta_1 ldots theta_m from the posterior distribution of iteration t-1 with weights w\nPerturb theta using a perturbation kernel\nRecompute the weights w (see (Toni et al, 2009) for details)\nSample m particles from pi: theta_1 ldots theta_m\nCompute the approximate distances by running the emulator regression: y_1 ldots y_m = textbfgpr(theta_1) ldots textbfgpr(theta_m)\nFor all j = 1 ldots m, if y_j leq varepsilon, then accept theta_j in the posterior sample\nAdvanced: details of the acceptance strategy can be tweaked. See GpABC.abc_select_emulated_particlesThis algorithm is implemented by Julia function EmulatedABCSMC."
},

{
    "location": "overview-abc/#References-1",
    "page": "ABC Parameter Inference",
    "title": "References",
    "category": "section",
    "text": "Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. Interface, (July 2008), 187–202. https://doi.org/10.1098/rsif.2008.0172"
},

{
    "location": "overview-ms/#",
    "page": "ABC Model Selection",
    "title": "ABC Model Selection",
    "category": "page",
    "text": ""
},

{
    "location": "overview-ms/#ms-overview-1",
    "page": "ABC Model Selection",
    "title": "ABC Model Selection Overview",
    "category": "section",
    "text": "The ABC SMC algorithm for model selection is available in full in the paper by Toni et al (see references for link). If using full model simulations the algorithm takes the following inputs:a prior over the models (the default in GpABC is a discrete uniform prior),\na schedule of thresholds for each ABC run (the first is rejection ABC and the subsequent runs are ABC SMC),\nparameter priors for each candidate model,\na maximum number of accepted particles per population, and\na maximum number of iterations per population (default 1000).As this is an ABC algorithm observed (reference) data, a distance metric and summary statistic must also be defined. As for other GpABC functions euclidean distance is the default distance metric.The pseudocode of the model selection algorithm isInitialise thresholds varepsilon_1varepsilon_T for Tx populations\nInitialise population index t=1\nWhile t leq T\nInitialise particle indicator i=1\nInitialise number of accepted particles for each of the M models A_1A_M=00\nWhile sum_m A_m  max no. of particles per population and i max no. of iterations per population\nSample model m from the model prior\nIf t = 1\nPerform rejection ABC for model m using a single particle using threshold varepsilon_t\nIf particle is accepted\nA_m = A_m + 1\nElse\nPerform ABC SMC for model m with a single particle using threshold varepsilon_t\nIf particle is accepted\nA_m = A_m + 1\ni = i + 1\nt = t + 1\nReturn number of accepted particles by each model at final population.Note that for model selection the number of accepted particles applies across all the models, with the model accepting the maximum number of particles in the final population being the one that is best supported by the data."
},

{
    "location": "overview-ms/#References-1",
    "page": "ABC Model Selection",
    "title": "References",
    "category": "section",
    "text": "Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. Interface, (July 2008), 187–202. https://doi.org/10.1098/rsif.2008.0172"
},

{
    "location": "overview-lna/#",
    "page": "LNA",
    "title": "LNA",
    "category": "page",
    "text": ""
},

{
    "location": "overview-lna/#lna-overview-1",
    "page": "LNA",
    "title": "Stochastic Inference (LNA) Overview",
    "category": "section",
    "text": "Biochemical reactions are stochastic in nature, and the distribution of stochastic simulation trajectories is generally non-Gaussian. To meet the Gaussian noise assumption of a GP and to consider computational efficiency, we employ the linear noise approximation (LNA), a first-order expansion of the Chemical Master Equation (CME) by decomposing the stochastic process into two ordinary differential equations (ODEs); one describing the evolution of the mean and the other describing the evolution of the covaraince of the trajectories textbfx(t) (i.e. mathbfx(t)sim mathcalN(phi(t)Sigma(t))):beginalign*\nfracdphi(t)dt=Scdot mathbfa(phi(t)) labelmean \nfracdSigma(t)dt=Scdot J cdot Sigma(t) + Sigma(t) cdot (Jcdot S)^T+\nOmega^-12 Scdot mathrmdiag mathbfa(phi(t)) cdot S^T labelcovar\nendalign*Here S is the stoichometry matrix of the system, textbfa is the reaction propensity vector. The J(t)_jk=partial a_jpartial phi_k is the Jacobian of the j^th reaction with respect to the k^th variable.These can be solved by numerical methods to describe how phi(t) (the mean) and Sigma(t) (the covariance) evolve with time. We can then draw samples from the above (time-dependent) multivariate Gaussian distribution and obtain realizations of stochastic simulation trajectories. Those trajectories can therefore be used for stochastic simulaiton based ABC. Please see Examples Section for more details."
},

{
    "location": "overview-lna/#References-1",
    "page": "LNA",
    "title": "References",
    "category": "section",
    "text": "Komorowski, M., Finkenstädt, B., Harper, C.V., and Rand, D.A. (2009). Bayesian inference of biochemical kinetic parameters using the linear noise approximation. BMC Bioinformatics, 10:343.\nSchnoerr, D., Sanguinetti, G., and Grima, R. (2017). Approximation and inference methods for stochastic biochemical kinetics—a tutorial review. Journal of Physics A: Mathematical and Theoretical, 50(9), 093001."
},

{
    "location": "overview-gp/#",
    "page": "Gaussian Process Regression",
    "title": "Gaussian Process Regression",
    "category": "page",
    "text": ""
},

{
    "location": "overview-gp/#gp-overview-1",
    "page": "Gaussian Process Regression",
    "title": "Gaussian Processes Regression Overview",
    "category": "section",
    "text": "An excellent introduction to Gaussian Processes is available in the book by (Rasmussen and Williams, 2006).Pages = [\"overview-gp.md\"]"
},

{
    "location": "overview-gp/#Gaussian-Process,-Prior-and-Posterior-1",
    "page": "Gaussian Process Regression",
    "title": "Gaussian Process, Prior and Posterior",
    "category": "section",
    "text": "A Gaussian Process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. This assumption is often referred to as the GP prior. In a regression setting, we are going to use GPs to approximate an unknown function f(x), x being a d-dimensional feature vector, x in mathbbR^d. We assume that our training data set contains of n points in mathbbR^d, and the test set - of m points in mathbbR^d. We denote the training data set as mathbfx mathbfx in mathbbR^n times d and the test data set as mathbfx^* mathbfx^* in mathbbR^m times d. Function values on training and test data sets are denoted as mathbfy = mathbff(x), and mathbfy^* = mathbff(x^*), respectively (in vectorised form). We also assume that the mean of the prior Gaussian distribution is zero, and its covariance matrix is known. Furthermore, we split the covariance matrix into the following regions:K: the covariance matrix computed on the training data, K in mathbbR^n times n\nK^**: the covariance matrix computed on the test data, K^** in mathbbR^m times m\nK^*: the covariance matrix between the training and test data, K^* in mathbbR^n times mIn this notation, the GP prior can be written asleft beginmatrix\nmathbfy\nmathbfy^*\nendmatrix right\nsim mathcalN left( 0\nleft beginmatrix\nK  K^*\nK^*top  K^**\nendmatrix right right)The desired approximation of f in mathbfx^* is the conditional distribution of mathbfy^*, given mathbfx, mathbfy and mathbfx^*. This distrubution, referred to as GP posterior, can be derived from the GP prior and the properties of a multivariate Normal distribution:beginalign*\nmathbfy^*  x y x^*  sim mathcalN(mathbftildey tildeK) \nmathbftildey  = K^*top K^-1 mathbfy \ntildeK  = K^** - K^*top K^-1 K^*\nendalign*mathbftildey and mathbftildeK are, respectively, the mean vector and the covariance matrix of the GP posterior. Often, we are not interested in non-diagonal elements of mathbftildeK. In such cases just the vector of diagonal elements is reported."
},

{
    "location": "overview-gp/#Kernels-and-Hyperparameters-1",
    "page": "Gaussian Process Regression",
    "title": "Kernels and Hyperparameters",
    "category": "section",
    "text": "We assume that the covariance between any two points x and x^ is given by a kernel function k(x x^), or in matrix notation, K_ij = k(mathbfx_i mathbfx_j). This kernel function is parameterised by a vector of hyperparameters mathbfeta = eta_1 ldots eta_p. The covariance matrix is thus also dependent on mathbfeta: K = K(mathbfeta).The optimal values of hyperparameters hateta can be obtained by finding the maximum value of log likelihood of the GP prior:beginalign*\nlog p(mathbfyeta) = -frac12mathbfy^top K^-1 mathbfy - frac12K - fracn2log(2pi) \nhateta = undersetmathbfetatextargmax(log p(mathbfymathbfeta))\nendalign*In GpABC this optimisation is performed using Optim package. By default, Conjugate Gradient bounded box optimisation is used, as long as the gradient with respect to hyperparameters is implemented for the kernel function. If the gradient implementation is not provided, Nelder Mead optimiser is used by default.The starting point of the optimisation can be specified by calling set_hyperparameters. If the starting point has not been provided, optimisation will start from all hyperparameters set to 1. Default upper and lower bounds are set to e^10 and e^10 , respectively, for each hyperparameter.For numerical stability the package uses logarithms of hyperparameters internally, when calling the log likelihood and kernel functions. Logarithmisation and exponentiation back takes place in gp_train function.The log likelihood function with log hyperparameters is implemented by gp_loglikelihood_log. This is the target function of the optimisation procedure in gp_train. There is also a version of log likelihood with actual (non-log) hyperparameters: gp_loglikelihood. The gradient of the log likelihood function with respect to logged hyperparameters is implemented by gp_loglikelihood_grad.Depending on the kernel, it is not uncommon for the log likelihood function to have multiple local optima. If a trained GP produces an unsatisfactory data fit, one possible workaround is trying to run gp_train several times with random starting points.Optim has a built in constraint of running no more than 1000 iterations of any optimisation algorithm. GpABC relies on this feature to ensure that the training procedure does not get stuck forever. As a consequence, the optimizer might exit prematurely, before reaching the local optimum. Setting log_level argument of gp_train to a value greater than zero will make it log its actions to standard output, including whether the local minimum has been reached or not.It is often convenient to model the measurement noise in the training data separately. This amounts to a normally distributed random variable being added to mathbfy. Denoting the variance of this random noise as sigma_n, this is equivalent to altering the covariance matrix to K_y = K + sigma_n I, where I is the identity matrix. Noise variance sigma_n is also a hyperparameter, that must be optimised with the rest of kernel hyperparameters. GpABC uses a joint hyperparameter vector, where sigma_n is always the last element."
},

{
    "location": "overview-gp/#Custom-Kernels-1",
    "page": "Gaussian Process Regression",
    "title": "Custom Kernels",
    "category": "section",
    "text": "GpABC ships with an extensible library of kernel functions. Each kernel is represented with a type that derives from AbstractGPKernel:SquaredExponentialIsoKernel\nSquaredExponentialArdKernel\nMaternIsoKernel\nMaternArdKernel\nExponentialIsoKernel\nExponentialArdKernelThese kernels rely on matrix of scaled squared distances between training/test inputs r_ij, which is computed by scaled_squared_distance function. The gradient vector of scaled squared distance derivatives with respect to length scale hyperparameter(s) is returned by scaled_squared_distance_grad function.The kernel covariance matrix is returned by function covariance. Optional speedups of this function covariance_diagonal and covariance_training are implemented for the pre-shipped kernels. The gradient with respect to log hyperparameters is computed by covariance_grad. The log_theta argument refers to the logarithms of kernel hyperparameters. Note that hyperparameters that do not affect the kernel (e.g. sigma_n ) are not included in log_theta.Custom kernels functions can be implemented  by adding more types that inherit from AbstractGPKernel."
},

{
    "location": "overview-gp/#References-1",
    "page": "Gaussian Process Regression",
    "title": "References",
    "category": "section",
    "text": "Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. ISBN 0-262-18253-X. http://www.gaussianprocess.org/gpml"
},

{
    "location": "summary_stats/#",
    "page": "Summary Statistics",
    "title": "Summary Statistics",
    "category": "page",
    "text": ""
},

{
    "location": "summary_stats/#summary_stats-1",
    "page": "Summary Statistics",
    "title": "Summary Statistics",
    "category": "section",
    "text": "The following summary statistics are available in GpABC:mean - mean of the data across the 2nd dimension\nvariance - variance of the data across the 2nd dimension\nmax - maximum of the data across the 2nd dimension\nmin - minimum of the data across the 2nd dimension\nrange - range of the data across the 2nd dimension (range = max - min)\nmedian - median of the data across the 2nd dimension\nq1 - the 1st quartile of the data across the 2nd dimension\nq3 - the 3rd quartile of the data across the 2nd dimension\niqr - the interquartile distance of the data across the 2nd dimension (iqr = q3 - q1)\nkeep_all - keep all data, reshaping it into a 1d vector"
},

{
    "location": "example-abc/#",
    "page": "ABC Parameter Inference",
    "title": "ABC Parameter Inference",
    "category": "page",
    "text": ""
},

{
    "location": "example-abc/#ABC-Example-1",
    "page": "ABC Parameter Inference",
    "title": "ABC Example",
    "category": "section",
    "text": "(Image: Binder) (Image: NBViewer)"
},

{
    "location": "example-ms/#",
    "page": "ABC Model Selection",
    "title": "ABC Model Selection",
    "category": "page",
    "text": ""
},

{
    "location": "example-ms/#Model-Selection-Example-1",
    "page": "ABC Model Selection",
    "title": "Model Selection Example",
    "category": "section",
    "text": "(Image: Binder) (Image: NBViewer)"
},

{
    "location": "example-lna/#",
    "page": "Stochastic inference (LNA)",
    "title": "Stochastic inference (LNA)",
    "category": "page",
    "text": ""
},

{
    "location": "example-lna/#Stochastic-Inference-(LNA)-Example-1",
    "page": "Stochastic inference (LNA)",
    "title": "Stochastic Inference (LNA) Example",
    "category": "section",
    "text": "(Image: Binder) (Image: NBViewer)"
},

{
    "location": "example-gp/#",
    "page": "Gaussian Processes",
    "title": "Gaussian Processes",
    "category": "page",
    "text": ""
},

{
    "location": "example-gp/#Gaussian-Processes-Examples-1",
    "page": "Gaussian Processes",
    "title": "Gaussian Processes Examples",
    "category": "section",
    "text": "(Image: Binder) (Image: NBViewer)"
},

{
    "location": "ref-abc/#",
    "page": "ABC Basic",
    "title": "ABC Basic",
    "category": "page",
    "text": ""
},

{
    "location": "ref-abc/#abc_ref-1",
    "page": "ABC Basic",
    "title": "ABC Basic Reference",
    "category": "section",
    "text": "GpABC functions for parameter estimation with Approximate Bayesian Computation. See also ABC Example."
},

{
    "location": "ref-abc/#Index-1",
    "page": "ABC Basic",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"ref-abc.md\"]"
},

{
    "location": "ref-abc/#GpABC.ABCRejectionOutput",
    "page": "ABC Basic",
    "title": "GpABC.ABCRejectionOutput",
    "category": "type",
    "text": "ABCRejectionOutput\n\nA container for the output of a rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_accepted::Int64: The number of accepted parameter vectors (particles) in the posterior.\nn_tries::Int64: The total number of parameter vectors (particles) that were tried.\nthreshold::Float64: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\npopulation::AbstractArray{Float64,2}: The parameter vectors (particles) in the posterior. Size: (n_accepted, n_params).\ndistances::AbstractArray{Float64,1}: The distances for each parameter vector (particle) in the posterior to the observed data in summary statistic space. Size: (n_accepted).\nweights::StatsBase.Weights: The weight of each parameter vector (particle) in the posterior.\n\n\n\n\n\n"
},

{
    "location": "ref-abc/#GpABC.ABCSMCOutput",
    "page": "ABC Basic",
    "title": "GpABC.ABCSMCOutput",
    "category": "type",
    "text": "ABCSMCOutput\n\nA container for the output of a rejection-ABC computation.\n\nFields\n\nn_params::Int64: The number of parameters to be estimated.\nn_accepted::Int64: The number of accepted parameter vectors (particles) in the posterior.\nn_tries::Int64: The total number of parameter vectors (particles) that were tried.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\npopulation::AbstractArray{Float64,2}: The parameter vectors (particles) in the posterior. Size: (n_accepted, n_params).\ndistances::AbstractArray{Float64,1}: The distances for each parameter vector (particle) in the posterior to the observed data in summary statistic space. Size: (n_accepted).\nweights::StatsBase.Weights: The weight of each parameter vector (particle) in the posterior.\n\n\n\n\n\n"
},

{
    "location": "ref-abc/#GpABC.SimulatedABCRejection",
    "page": "ABC Basic",
    "title": "GpABC.SimulatedABCRejection",
    "category": "function",
    "text": "SimulatedABCRejection(\n    reference_data,\n    simulator_function,\n    priors,\n    threshold,\n    n_particles;\n    summary_statistic   = \"keep_all\",\n    distance_function   = Distances.euclidean,\n    max_iter            = 10 * n_particles,\n    write_progress      = true,\n    progress_every      = 1000,\n    )\n\nRun simulation-based rejection ABC algorithm. Particles are sampled from the prior, and the model is simulated for each particle. Only those particles are included in the posterior that have distance between the simulation results and the reference data below the threshold (after taking summary statistics into account).\n\nSee ABC Overview for more details.\n\nMandatory arguments\n\nreference_data::AbstractArray{Float,2}: Observed data to which the simulated model output will be compared. Array dimensions sould match that of the simulator function result.\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: Continuous univariate distributions, from which candidate parameters will be sampled. Array size should match the number of parameters.\nthreshold::Float: The varepsilon threshold to be used in ABC algorithm. Only those particles that produce simulated results that are within this threshold from the reference data are included into the posterior.\nn_particles::Int: The number of parameter vectors (particles) that will be included in the posterior.\n\nOptional keyword arguments\n\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Summary statistics that will be applied to the data before computing the distances. Defaults to keep_all. See detailed documentation of summary statistics.\ndistance_function::Function: A function that will be used to compute the distance between the summary statistic of the simulated data and that of reference data. Defaults to Distances.euclidean.\nmax_iter::Int: The maximum number of simulations that will be run. The default is 1000 * n_particles.\nwrite_progress::Bool: Whether algorithm progress should be printed on standard output. Defaults to true.\nprogress_every::Int: Number of iterations at which to print progress. Defaults to 1000.\n\nReturns\n\nAn ABCRejectionOutput object.\n\n\n\n\n\n"
},

{
    "location": "ref-abc/#GpABC.EmulatedABCRejection",
    "page": "ABC Basic",
    "title": "GpABC.EmulatedABCRejection",
    "category": "function",
    "text": "EmulatedABCRejection(\n    reference_data,\n    simulator_function,\n    priors,\n    threshold,\n    n_particles,\n    n_design_points;\n    summary_statistic               = \"keep_all\",\n    distance_function               = Distances.euclidean,\n    batch_size                      = 10*n_particles,\n    max_iter                        = 1000,\n    emulator_training               = DefaultEmulatorTraining(),\n    emulated_particle_selection     = MeanEmulatedParticleSelection(),\n    write_progress                  = true,\n    progress_every                  = 1000,\n    )\n\nRun emulation-based rejection ABC algorithm. Model simulation results are used to train the emulator, which is then used to get the approximated distance for each particle. The rest of the workflow is similar to SimulatedABCRejection.\n\nSee ABC Overview for more details.\n\nMandatory arguments\n\nreference_data::AbstractArray{Float,2}: Observed data to which the simulated model output will be compared. Array dimensions sould match that of the simulator function result.\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: Continuous univariate distributions, from which candidate parameters will be sampled. Array size should match the number of parameters.\nthreshold::Float: The varepsilon threshold to be used in ABC algorithm. Only those particles that produce emulated results that are within this threshold from the reference data are included into the posterior.\nn_particles::Int: The number of parameter vectors (particles) that will be included in the posterior.\nn_design_points::Int: The number of design particles that will be simulated to traing the emulator.\n\nOptional keyword arguments\n\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Summary statistics that will be applied to the data before computing the distances. Defaults to keep_all. See detailed documentation of summary statistics.\ndistance_function::Function: A function that will be used to compute the distance between the summary statistic of the simulated data and that of reference data. Defaults to Distances.euclidean.\nbatch_size::Int: The number of particles that will be emulated on each iteration. Defaults to 1000 * n_particles.\nmax_iter::Int: The maximum number of emulations that will be run. Defaults to 1000.\nemulator_training<:AbstractEmulatorTraining: This determines how the emulator will be trained. See AbstractEmulatorTraining for more details.\nemulated_particle_selection<:AbstractEmulatedParticleSelection: This determines how the particles that will be added to the posterior are selected after each emulation run. See AbstractEmulatedParticleSelection for details. Defaults to MeanEmulatedParticleSelection.\nwrite_progress::Bool: Whether algorithm progress should be printed on standard output. Defaults to true.\n\nReturns\n\nAn ABCRejectionOutput object.\n\n\n\n\n\n"
},

{
    "location": "ref-abc/#GpABC.SimulatedABCSMC",
    "page": "ABC Basic",
    "title": "GpABC.SimulatedABCSMC",
    "category": "function",
    "text": "SimulatedABCSMC(\n    reference_data,\n    simulator_function,\n    priors,\n    threshold_schedule,\n    n_particles;\n    summary_statistic   = \"keep_all\",\n    distance_function   = Distances.euclidean,\n    max_iter            = 10 * n_particles,\n    write_progress      = true,\n    progress_every      = 1000,\n    )\n\nRun a simulation-based ABC-SMC algorithm. This is similar to SimulatedABCRejection, the main difference being that an array of thresholds is provided instead of a single threshold. It is assumed that thresholds are sorted in decreasing order.\n\nA simulation based ABC iteration is executed for each threshold. For the first threshold, the provided prior is used. For each subsequent threshold, the posterior from the previous iteration is used as a prior.\n\nSee ABC Overview for more details.\n\nMandatory arguments\n\nreference_data::AbstractArray{Float,2}: Observed data to which the simulated model output will be compared. Array dimensions sould match that of the simulator function result.\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: Continuous univariate distributions, from which candidate parameters will be sampled during the first iteration. Array size should match the number of parameters.\nthreshold_schedule::AbstractArray{Float,1}: The threshold schedule to be used in ABC algorithm. An ABC iteration is executed for each threshold. It is assumed that thresholds are sorted in decreasing order.\nn_particles::Int: The number of parameter vectors (particles) that will be included in the posterior.\n\nOptional keyword arguments\n\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Summary statistics that will be applied to the data before computing the distances. Defaults to keep_all. See detailed documentation of summary statistics.\ndistance_function::Function: A function that will be used to compute the distance between the summary statistic of the simulated data and that of reference data. Defaults to Distances.euclidean.\nmax_iter::Int: The maximum number of simulations that will be run. The default is 1000 * n_particles.\nwrite_progress::Bool: Whether algorithm progress should be printed on standard output. Defaults to true.\nprogress_every::Int: Number of iterations at which to print progress. Defaults to 1000.\n\nReturns\n\nAn ABCSMCOutput object.\n\n\n\n\n\n"
},

{
    "location": "ref-abc/#GpABC.EmulatedABCSMC",
    "page": "ABC Basic",
    "title": "GpABC.EmulatedABCSMC",
    "category": "function",
    "text": "EmulatedABCSMC(\n    reference_data,\n    simulator_function,\n    priors,\n    threshold_schedule,\n    n_particles,\n    n_design_points;\n    summary_statistic               = \"keep_all\",\n    distance_function               = Distances.euclidean,\n    batch_size                      = 10*n_particles,\n    max_iter                        = 1000,\n    emulator_training               = DefaultEmulatorTraining(),\n    emulator_retraining             = NoopRetraining(),\n    emulated_particle_selection     = MeanEmulatedParticleSelection(),\n    write_progress                  = true,\n    progress_every                  = 1000,\n    )\n\nRun emulation-based ABC-SMC algorithm. This is similar to EmulatedABCRejection, the main difference being that an array of thresholds is provided instead of a single threshold. It is assumed that thresholds are sorted in decreasing order.\n\nAn emulation based ABC iteration is executed for each threshold. For the first threshold, the provided prior is used. For each subsequent threshold, the posterior from the previous iteration is used as a prior.\n\nSee ABC Overview for more details.\n\nMandatory arguments\n\nreference_data::AbstractArray{Float,2}: Observed data to which the simulated model output will be compared. Array dimensions sould match that of the simulator function result.\nsimulator_function::Function: A function that takes a parameter vector as an argument and outputs model results.\npriors::AbstractArray{ContinuousUnivariateDistribution,1}: Continuous univariate distributions, from which candidate parameters will be sampled during the first iteration. Array size should match the number of parameters.\nthreshold_schedule::AbstractArray{Float,1}: The threshold schedule to be used in ABC algorithm. An ABC iteration is executed for each threshold. It is assumed that thresholds are sorted in decreasing order.\nn_particles::Int: The number of parameter vectors (particles) that will be included in the posterior.\nn_design_points::Int: The number of design particles that will be simulated to traing the emulator.\n\nOptional keyword arguments\n\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Summary statistics that will be applied to the data before computing the distances. Defaults to keep_all. See detailed documentation of summary statistics.\ndistance_function::Function: A function that will be used to compute the distance between the summary statistic of the simulated data and that of reference data. Defaults to Distances.euclidean.\nbatch_size::Int: The number of particles that will be emulated on each iteration. Defaults to 1000 * n_particles.\nmax_iter::Int: The maximum number of emulations that will be run. Defaults to 1000.\nemulator_training<:AbstractEmulatorTraining: This determines how the emulator will be trained for each iteration. See AbstractEmulatorTraining for more details.\nemulator_retraining<:AbstractEmulatorRetraining: This is used to specify parameters of additional emulator retraining that can be done for each iteration. By default this retraining is switched off (NoopRetraining). See [AbstractEmulatorRetraining] for more details.\nemulated_particle_selection<:AbstractEmulatedParticleSelection: This determines how the particles that will be added to the posterior are selected after each emulation run. See AbstractEmulatedParticleSelection for details. Defaults to MeanEmulatedParticleSelection.\nwrite_progress::Bool: Whether algorithm progress should be printed on standard output. Defaults to true.\n\nReturns\n\nAn ABCSMCOutput object.\n\n\n\n\n\n"
},

{
    "location": "ref-abc/#Types-and-Functions-1",
    "page": "ABC Basic",
    "title": "Types and Functions",
    "category": "section",
    "text": "ABCRejectionOutput\nABCSMCOutput\nSimulatedABCRejection\nEmulatedABCRejection\nSimulatedABCSMC\nEmulatedABCSMC"
},

{
    "location": "ref-abc-advanced/#",
    "page": "ABC Advanced",
    "title": "ABC Advanced",
    "category": "page",
    "text": ""
},

{
    "location": "ref-abc-advanced/#ABC-Advanced-Reference-1",
    "page": "ABC Advanced",
    "title": "ABC Advanced Reference",
    "category": "section",
    "text": "Advanced aspects of GpABC support for parameter estimation with Approximate Bayesian Computation."
},

{
    "location": "ref-abc-advanced/#Index-1",
    "page": "ABC Advanced",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"ref-abc-advanced.md\"]"
},

{
    "location": "ref-abc-advanced/#GpABC.AbstractEmulatorTraining",
    "page": "ABC Advanced",
    "title": "GpABC.AbstractEmulatorTraining",
    "category": "type",
    "text": "AbstractEmulatorTraining\n\nSubtypes of this abstract type control how the emulator is trained for emulation-based ABC algorithms (rejection and SMC). At the moment, only DefaultEmulatorTraining is shipped. Custom emulator training procedure can be implemented by creating new subtypes of this type and overriding train_emulator for them.\n\nA typical use case would be trying to control the behaviour of gp_train more tightly, or not using it altogeather (e.g. using another optimisation package).\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.DefaultEmulatorTraining",
    "page": "ABC Advanced",
    "title": "GpABC.DefaultEmulatorTraining",
    "category": "type",
    "text": "DefaultEmulatorTraining <: AbstractEmulatorTraining\n\nFields\n\nkernel::AbstractGPKernel: the kernel (AbstractGPKernel) that will be used with the Gaussian Process (GPModel). Defaults to SquaredExponentialArdKernel.\n\ntrain_emulator method with this argument type calls gp_train with default arguments.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.AbstractEmulatedParticleSelection",
    "page": "ABC Advanced",
    "title": "GpABC.AbstractEmulatedParticleSelection",
    "category": "type",
    "text": "AbstractEmulatedParticleSelection\n\nSubtypes of this type control the criteria that determine what particles are included in the posterior for emulation-based ABC. Custom strategies can be implemented by creating new subtypes of this type and overriding abc_select_emulated_particles for them.\n\nThree implementations are shipped:\n\nMeanEmulatedParticleSelection\nMeanVarEmulatedParticleSelection\nPosteriorSampledEmulatedParticleSelection\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.MeanVarEmulatedParticleSelection",
    "page": "ABC Advanced",
    "title": "GpABC.MeanVarEmulatedParticleSelection",
    "category": "type",
    "text": "MeanVarEmulatedParticleSelection <: AbstractEmulatedParticleSelection\n\nWhen this strategy is used, the particles for which both mean and standard deviation returned by gp_regression is below the ABC threshold are included in the posterior. In pseudocode:\n\nmeans, vars = gp_regression(parameters, gpm)\naccepted_indices = findall((means .<= threshold) .& (sqrt.(vars) .<= threshold))\n\nThe rationale behind using this selection strategy is to take into account the \"level of uncertainty\" about the regression prediction that is provided by the Gaussian Process in form of standard deviation. So, even if the mean of the GP is below the threshold, but the GP is \"uncertain\" about it (i.e. the variance is high), this particle will not be included in the posterior distribution of ABC. It is a more stringent acceptance criteria than MeanEmulatedParticleSelection.\n\nFields\n\nvariance_threshold_factor: scaling factor, by which the ABC threshold is multiplied\n\nbefore checking the standard deviation. Defaults to 1.0.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.MeanEmulatedParticleSelection",
    "page": "ABC Advanced",
    "title": "GpABC.MeanEmulatedParticleSelection",
    "category": "type",
    "text": "MeanEmulatedParticleSelection <: AbstractEmulatedParticleSelection\n\nWhen this strategy is used, the particles for which only the mean value returned by gp_regression is below the ABC threshold are included in the posterior. Variance is not checked.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.PosteriorSampledEmulatedParticleSelection",
    "page": "ABC Advanced",
    "title": "GpABC.PosteriorSampledEmulatedParticleSelection",
    "category": "type",
    "text": "PosteriorSampledEmulatedParticleSelection <: AbstractEmulatedParticleSelection\n\nWhen this strategy is used, the distance is sampled from the GP posterior of the gp_regression object. If the sampled distance is below the threshold the particle is accepted.\n\nFields\n\nuse_diagonal_covariance: if true, the GP posterior covariance will be approximated by its\n\ndiagonal elements only. Defaults to false.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.AbstractEmulatorRetraining",
    "page": "ABC Advanced",
    "title": "GpABC.AbstractEmulatorRetraining",
    "category": "type",
    "text": "AbstractEmulatorRetraining\n\nSubtypes of this abstract type control the additional retraining procedure that may or may not be carried out before each iteration of emulation-based ABC. Custom strategies may be implemented by creating new subtypes of this type and new abc_retrain_emulator methods for them.\n\nThe following implementations are shipped:\n\nIncrementalRetraining\nPreviousPopulationRetraining\nPreviousPopulationThresholdRetraining\nNoopRetraining\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.IncrementalRetraining",
    "page": "ABC Advanced",
    "title": "GpABC.IncrementalRetraining",
    "category": "type",
    "text": "IncrementalRetraining <: AbstractEmulatorRetraining\n\nThis emulator retraining strategy samples extra particles from the previous population, and adds them to the set of design points that were used on the previous iteration. The new design points are filtered according to the threshold. The combined set of design points is used to train the new emulator.\n\nFields\n\ndesign_points: number of design points to add on each iteration\nmax_simulations: maximum number of simulations to perform during re-trainging on each iteration\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.PreviousPopulationRetraining",
    "page": "ABC Advanced",
    "title": "GpABC.PreviousPopulationRetraining",
    "category": "type",
    "text": "PreviousPopulationRetraining <: AbstractEmulatorRetraining\n\nThis emulator retraining strategy samples extra particles from the previous population, and uses them to re-train the emulator from scratch. No filtering of the new design points is performed. Design points from the previous iteration are discarded.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.PreviousPopulationThresholdRetraining",
    "page": "ABC Advanced",
    "title": "GpABC.PreviousPopulationThresholdRetraining",
    "category": "type",
    "text": "PreviousPopulationThresholdRetraining <: AbstractEmulatorRetraining\n\nThis emulator retraining strategy samples extra particles from the previous population, and uses them to re-train the emulator from scratch. Design points from the previous iteration are discarded. This strategy allows to control how many design points are sampled with distance to the reference data below the threshold.\n\nFields:\n\nn_design_points: number of design points\nn_below_threshold: number of design points below the threshold\nmax_iter: maximum number of simulations to perform on each re-training iteration\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.NoopRetraining",
    "page": "ABC Advanced",
    "title": "GpABC.NoopRetraining",
    "category": "type",
    "text": "NoopRetraining <: AbstractEmulatorRetraining\n\nA sentinel retraining strategy that does not do anything. When used, the emulator is trained only once at the start of the process.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.train_emulator",
    "page": "ABC Advanced",
    "title": "GpABC.train_emulator",
    "category": "function",
    "text": "train_emulator(training_x, training_y, emulator_training<:AbstractEmulatorTraining)\n\nTrain the emulator. For custom training procedure, a new subtype of AbstractEmulatorTraining should be created, and a method of this function should be created for it.\n\nReturns\n\nA trained emulator. Shipped implementations return a GPModel.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.abc_retrain_emulator",
    "page": "ABC Advanced",
    "title": "GpABC.abc_retrain_emulator",
    "category": "function",
    "text": "abc_retrain_emulator(\n    gp_model,\n    particle_sampling_function,\n    epsilon,\n    training_input::EmulatorTrainingInput,\n    retraining_settings<:AbstractEmulatorRetraining\n    )\n\nAdditional retraining procedure for the emulator that may or may not be executed before every iteration of emulation based ABC. Details of the procedure are determined by the subtype of AbstractEmulatorRetraining that is passed as an argument.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#GpABC.abc_select_emulated_particles",
    "page": "ABC Advanced",
    "title": "GpABC.abc_select_emulated_particles",
    "category": "function",
    "text": "abc_select_emulated_particles(emulator, particles, threshold, selection<:AbstractEmulatedParticleSelection)\n\nCompute the approximate distances by running the regression-based emulator on the provided particles, and return the accepted particles. Acceptance strategy is determined by selection - subtype of AbstractEmulatedParticleSelection.\n\nArguments\n\ngpm: the GPModel, that contains the training data (x and y), the kernel, the hyperparameters and the test data for running the regression.\nparameters: array of parameters to test (the particles).\nthreshold: if the distance for a particle is below threshold then it is accepted as a posterior sample.\nselection: the acceptance strategy (a subtype of AbstractEmulatedParticleSelection). This determines the method by which an emulated distance is accepted.\n\nReturn\n\nA tuple of accepted distances, and indices of the accepted particles in the supplied particles array.\n\n\n\n\n\n"
},

{
    "location": "ref-abc-advanced/#Types-and-Functions-1",
    "page": "ABC Advanced",
    "title": "Types and Functions",
    "category": "section",
    "text": "AbstractEmulatorTraining\nDefaultEmulatorTraining\nAbstractEmulatedParticleSelection\nMeanVarEmulatedParticleSelection\nMeanEmulatedParticleSelection\nPosteriorSampledEmulatedParticleSelection\nAbstractEmulatorRetraining\nIncrementalRetraining\nPreviousPopulationRetraining\nPreviousPopulationThresholdRetraining\nNoopRetraining\nGpABC.train_emulator\nGpABC.abc_retrain_emulator\nGpABC.abc_select_emulated_particles"
},

{
    "location": "ref-lna/#",
    "page": "Stochastic inference (LNA)",
    "title": "Stochastic inference (LNA)",
    "category": "page",
    "text": ""
},

{
    "location": "ref-lna/#LNA-Reference-1",
    "page": "Stochastic inference (LNA)",
    "title": "LNA Reference",
    "category": "section",
    "text": "GpABC functions for Linear Noise Approximation."
},

{
    "location": "ref-lna/#Index-1",
    "page": "Stochastic inference (LNA)",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"ref-lna.md\"]"
},

{
    "location": "ref-lna/#GpABC.LNA",
    "page": "Stochastic inference (LNA)",
    "title": "GpABC.LNA",
    "category": "type",
    "text": "LNA\n\nThis is a structure which will hold the LNA: the mean of the trajectories and the covariance between the species.\n\nArguments\n\ntraj_means: A (number of species) x (number of time points) array which holds the mean trajectory for each species on each row of the array.\ntraj_covars: An array which holds the covariance matrix of the species at each time point.\ntime_points: The timepoints the system was solved for.\n\n\n\n\n\n"
},

{
    "location": "ref-lna/#GpABC.LNAInput",
    "page": "Stochastic inference (LNA)",
    "title": "GpABC.LNAInput",
    "category": "type",
    "text": "LNAInput\n\nThis is a structure which holds the inputs needed for computing the Linear Noise Approximation (LNA). This structure will hold the stochastic system as provided by the user; uniquely defined through kinetic parameters, the rates of the system and the stoichiometry matrix.\n\nArguments\n\nparams::AbstractArray{Float64,1}: The rate parameters of the stochastic model.\nS::AbstractArray{Float64,2}: the stochiometry matrix of the system. Size: number of reactions x number of species.\nreaction_rate_function::Function,: This is a function f(x, parameters) which should return an array of the reaction rates of the system, i.e. S*f would describe the ODE representation of the system.\nvolume::Float64: The volume of the reactants of the system.\n\n\n\n\n\n"
},

{
    "location": "ref-lna/#GpABC.compute_LNA",
    "page": "Stochastic inference (LNA)",
    "title": "GpABC.compute_LNA",
    "category": "function",
    "text": "compute_LNA(input::LNAInput,\n    x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},\n    Tspan::Tuple{Float64,Float64},\n    saveat::Float64,\n    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();\n    kwargs...)\n\nThe function computes the linear noise approximation to system through construction of two ODEs: one describing the trajectories of the mean of the LNA and the other describing the change the covariance between the variables. These outputs are held in a LNA structure.\n\nArguments\n\nx0::Tuple{AbstractArray{Float64,2},AbstractArray{Float64,2}}: The initial conditions of the system. In the form of (the initial conditions of the species, the initial covariance matrix of the system).\nTspan::Tuple{Float64,Float64}: The start and end times of the simulation.\nsaveat::Float64: The number of time points the use wishes to solve the system for.\nsolver::DEAlgorithm: The ODE solver the user wishes to use, for example OrdinaryDiffEq.RK4().\n\n#Returns\n\nLNA\n\n\n\n\n\n"
},

{
    "location": "ref-lna/#GpABC.get_LNA_trajectories",
    "page": "Stochastic inference (LNA)",
    "title": "GpABC.get_LNA_trajectories",
    "category": "function",
    "text": "get_LNA_trajectories(input::LNAInput, n_samples::Int64,\n    x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},\n    Tspan::Tuple{Float64,Float64},\n    saveat::Float64,\n    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();\n    kwargs...)\n\nA function which computes the LNA and then samples from the it to output sampled trajectories. The user can also sample more than one trajectory; which are then averaged.\n\nArguments\n\ninput::LNAInput: LNAInput stucture.\nn_samples::Int64: The number of sampled tracjectories to be sampled and then averaged.\nx0::Tuple{AbstractArray{Float64,2},AbstractArray{Float64,2}}: The initial conditions of the system. In the form of (the initial conditions of the species, the initial covariance matrix of the system).\nTspan::Tuple{Float64,Float64}: The start and end times of the simulation.\nsaveat::Float64: The number of time points the use wishes to solve the system for.\nsolver::DEAlgorithm: The ODE solver the user wishes to use, for example OrdinaryDiffEq.RK4() .\n\n#Returns\n\nA (number of species) x (number of time points) array which holds the averaged trajectory for each species on each row of the array.\n\n\n\n\n\n"
},

{
    "location": "ref-lna/#GpABC.sample_LNA_trajectories-Tuple{LNA,Int64}",
    "page": "Stochastic inference (LNA)",
    "title": "GpABC.sample_LNA_trajectories",
    "category": "method",
    "text": "sample_LNA_trajectories(lna::LNA, n_samples::Int64)\n\nA function which samples from the LNA to output sampled trajectories. The LNA gives the mean of the tracjectories and the covariance between them; hence a single trajectory can be sampled from a Multivariate Normal distribution. The user can also sample more than one trajectory; which are then averaged.\n\nArguments\n\nlna::LNA: LNA stucture.\nn_samples::Int64: The number of sampled tracjectories to be sampled and then averaged.\n\n#Returns\n\nA (number of species) x (number of time points) array which holds the averaged trajectory for each species on each row of the array.\n\n\n\n\n\n"
},

{
    "location": "ref-lna/#Types-and-Functions-1",
    "page": "Stochastic inference (LNA)",
    "title": "Types and Functions",
    "category": "section",
    "text": "Modules = [GpABC]\nPages = [\"lna.jl\"]"
},

{
    "location": "ref-ms/#",
    "page": "Model Selection",
    "title": "Model Selection",
    "category": "page",
    "text": ""
},

{
    "location": "ref-ms/#Model-Selection-Reference-1",
    "page": "Model Selection",
    "title": "Model Selection Reference",
    "category": "section",
    "text": "GpABC functions for Model Selection."
},

{
    "location": "ref-ms/#Index-1",
    "page": "Model Selection",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"ref-ms.md\"]"
},

{
    "location": "ref-ms/#GpABC.SimulatedModelSelection",
    "page": "Model Selection",
    "title": "GpABC.SimulatedModelSelection",
    "category": "function",
    "text": "SimulatedModelSelection\n\nPerform model selection using simulation-based ABC.\n\nArguments\n\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (nmodeltrajectories, ntimepoints)\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\nparameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution},1}: Priors for the parameters of each model. The length of the outer array is the number of models.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. Defaults to keep_all. See detailed documentation of summary statistics.\nsimulator_functions::AbstractArray{Function,1}: An array of functions that take a parameter vector as an argument and outputs model results (one per model).\n\'model_prior::DiscreteUnivariateDistribution\': The prior from which models are sampled. Default is a discrete, uniform distribution.\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).\nmax_iter::Int: The maximum number of simulations that will be run. The default is 1000*n_particles. Each iteration samples a single model and performs ABC using a single particle.\n\nReturns\n\nA ModelSelectionOutput object that contains which models are supported by the observed data.\n\n\n\n\n\n"
},

{
    "location": "ref-ms/#GpABC.EmulatedModelSelection",
    "page": "Model Selection",
    "title": "GpABC.EmulatedModelSelection",
    "category": "function",
    "text": "EmulatedModelSelection\n\nPerform model selection using emulation-based ABC.\n\nArguments\n\nn_design_points::Int64: The number of parameter vectors used to train the Gaussian process emulator.\nreference_data::AbstractArray{Float64,2}: The observed data to which the simulated model output will be compared. Size: (nmodeltrajectories, ntimepoints)\nn_particles::Int64: The number of parameter vectors (particles) that will be included in the final posterior.\nthreshold_schedule::AbstractArray{Float64}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.\nparameter_priors::AbstractArray{AbstractArray{ContinuousUnivariateDistribution},1}: Priors for the parameters of each model. The length of the outer array is the number of models.\nsummary_statistic::Union{String,AbstractArray{String,1},Function}: Either: 1. A String or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. Defaults to keep_all. See detailed documentation of summary statistics.\nsimulator_functions::AbstractArray{Function,1}: An array of functions that take a parameter vector as an argument and outputs model results (one per model).\n\'model_prior::DiscreteUnivariateDistribution\': The prior from which models are sampled. Default is a discrete, uniform distribution.\ndistance_function::Function: Any function that computes the distance between 2 1D Arrays. Optional argument (default is to use the Euclidean distance).\nmax_iter::Int: The maximum number of simulations that will be run. The default is 1000*n_particles. Each iteration samples a single model and performs ABC using a single particle.\nmax_batch_size::Int: The maximum batch size for the emulator when making predictions.\n\nReturns\n\nA ModelSelectionOutput object that contains which models are supported by the observed data.\n\n\n\n\n\n"
},

{
    "location": "ref-ms/#GpABC.ModelSelectionOutput",
    "page": "Model Selection",
    "title": "GpABC.ModelSelectionOutput",
    "category": "type",
    "text": "ModelSelectionOutput\n\nContains results of a model selection computation, including which models are best supported by the observed data and the parameter poseteriors at each population for each model.\n\nFields\n\nM::Int64: The number of models.\nn_accepted::AbstractArray{AbstractArray{Int64,1},1}: The number of parameters accepted by each model at each population. n_accepted[i][j] contains the number of acceptances for model j at population i.\nthreshold_schedule::AbstractArray{Float64,1}: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.\nsmc_outputs::AbstractArray{ABCSMCOutput,1}: A [\'SimulatedABCSMCOutput\']@(ref) or [\'EmulatedABCSMCOutput\']@(ref) for each model. Use to find details of the ABC results at each population.\ncompleted_all_populations::Bool: Indicates whether the algorithm completed all the populations successfully. A successful population is one where at least one model accepts at least one particle.\n\n\n\n\n\n"
},

{
    "location": "ref-ms/#Types-and-Functions-1",
    "page": "Model Selection",
    "title": "Types and Functions",
    "category": "section",
    "text": "SimulatedModelSelection\nEmulatedModelSelection\nModelSelectionOutput"
},

{
    "location": "ref-gp/#",
    "page": "Gaussian Processes",
    "title": "Gaussian Processes",
    "category": "page",
    "text": ""
},

{
    "location": "ref-gp/#Gaussian-Processes-Reference-1",
    "page": "Gaussian Processes",
    "title": "Gaussian Processes Reference",
    "category": "section",
    "text": "GpABC functions for Gaussian Process Regression. See also Gaussian Processes Overview, Gaussian Processes Examples."
},

{
    "location": "ref-gp/#Index-1",
    "page": "Gaussian Processes",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"ref-gp.md\"]"
},

{
    "location": "ref-gp/#GpABC.GPModel",
    "page": "Gaussian Processes",
    "title": "GpABC.GPModel",
    "category": "type",
    "text": "GPModel\n\nThe main type that is used by most functions within the package.\n\nAll data matrices are row-major.\n\nFields\n\nkernel::AbstractGPKernel: the kernel\ngp_training_x::AbstractArray{Float64, 2}: training x. Size: n times d.\ngp_training_y::AbstractArray{Float64, 2}: training y. Size: n times 1.\ngp_test_x::AbstractArray{Float64, 2}: test x.  Size: m times d.\ngp_hyperparameters::AbstractArray{Float64, 1}: kernel hyperparameters, followed by standard deviation of intrinsic noise sigma_n, which is always the last element in the array.\ncache::HPOptimisationCache: cache of matrices that can be re-used between calls to gp_loglikelihood and gp_loglikelihood_grad\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.GPModel",
    "page": "Gaussian Processes",
    "title": "GpABC.GPModel",
    "category": "type",
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nDefault constructor of GPModel, that will use SquaredExponentialIsoKernel. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.GPModel",
    "page": "Gaussian Processes",
    "title": "GpABC.GPModel",
    "category": "type",
    "text": "GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},\n        kernel::AbstractGPKernel\n        [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])\n\nConstructor of GPModel that allows the kernel to be specified. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.GPModel-Tuple{}",
    "page": "Gaussian Processes",
    "title": "GpABC.GPModel",
    "category": "method",
    "text": "GPModel(;training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),\n    kernel::AbstractGPKernel=SquaredExponentialIsoKernel(),\n    gp_hyperparameters::AbstractArray{Float64, 1}=Array{Float64}(0))\n\nConstructor of GPModel with explicit arguments. Arguments that are passed as 1-d vectors will be reshaped into 2-d.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_loglikelihood-Tuple{AbstractArray{Float64,1},GPModel}",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_loglikelihood",
    "category": "method",
    "text": "gp_loglikelihood(theta::AbstractArray{Float64, 1}, gpm::GPModel)\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_loglikelihood-Tuple{GPModel}",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_loglikelihood",
    "category": "method",
    "text": "gp_loglikelihood(gpm::GPModel)\n\nCompute the log likelihood function, based on the kernel and training data specified in gpm.\n\nlog p(y vert X theta) = - frac12(y^TK^-1y + log vert K vert + n log 2 pi)\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_loglikelihood_grad-Tuple{AbstractArray{Float64,1},GPModel}",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_loglikelihood_grad",
    "category": "method",
    "text": "gp_loglikelihood_grad(theta::AbstractArray{Float64, 1}, gpem::GPModel)\n\nGradient of the log likelihood function (gp_loglikelihood_log) with respect to logged hyperparameters.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_loglikelihood_log-Tuple{AbstractArray{Float64,1},GPModel}",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_loglikelihood_log",
    "category": "method",
    "text": "gp_loglikelihood_log(theta::AbstractArray{Float64, 1}, gpm::GPModel)\n\nLog likelihood function with log hyperparameters. This is the target function of the hyperparameters optimisation procedure. Its gradient is coputed by gp_loglikelihood_grad.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_regression-Tuple{GPModel}",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_regression",
    "category": "method",
    "text": "gp_regression(gpm::GPModel; <optional keyword arguments>)\n\nRun the Gaussian Process Regression.\n\nArguments\n\ngpm: the GPModel, that contains the training data (x and y), the kernel, the hyperparameters and the test data for running the regression.\ntest_x: if specified, overrides the test x in gpm. Size m times d.\nlog_level::Int (optional): log level. Default is 0, which is no logging at all. 1 makes gp_regression print basic information to standard output.\nfull_covariance_matrix::Bool (optional): whether we need the full covariance matrix, or just the variance vector. Defaults to false (i.e. just the variance).\nbatch_size::Int (optional): If full_covariance_matrix is set to false, then the mean and variance vectors will be computed in batches of this size, to avoid allocating huge matrices. Defaults to 1000.\nobservation_noise::Bool (optional): whether the observation noise (with variance sigma_n^2) should be included in the output variance. Defaults to true.\n\nReturn\n\nA tuple of (mean, var). mean is a mean vector of the output multivariate Normal distribution, of size m. var is either the covariance matrix of size m times m, or a variance vector of size m, depending on full_covariance_matrix flag.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_regression-Tuple{Union{AbstractArray{Float64,1}, AbstractArray{Float64,2}},GPModel}",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_regression",
    "category": "method",
    "text": "gp_regression(test_x::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}},\n    gpem::GPModel; <optional keyword arguments>)\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_regression_sample",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_regression_sample",
    "category": "function",
    "text": "gp_regression_sample(test_x::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}}, n_samples::Int64, gpem::GPModel)\n\nReturn n_samples random samples from the Gaussian process posterior, evaluated at test_x. gp_regression.\n\nArguments\n\ntest_x: if specified, overrides the test x in gpm. Size m times d.\nn_samples: integer specifying the number of posterior samples.\ngpm: the GPModel, that contains the training data (x and y), the kernel, the hyperparameters and the test data for running the regression.\nfull_cov_matrix: whether to use the full covariance matrix or just its diagonal elements (default true).\n\nReturn\n\nAn array of posterior samples with shape m times n_samples if n_samples>1 and m otherwise.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.set_hyperparameters-Tuple{GPModel,AbstractArray{Float64,1}}",
    "page": "Gaussian Processes",
    "title": "GpABC.set_hyperparameters",
    "category": "method",
    "text": "set_hyperparameters(gpm::GPModel, hypers::AbstractArray{Float64, 1})\n\nSet the hyperparameters of the GPModel\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#GpABC.gp_train-Union{Tuple{GPModel}, Tuple{TOpt}} where TOpt<:Optim.AbstractOptimizer",
    "page": "Gaussian Processes",
    "title": "GpABC.gp_train",
    "category": "method",
    "text": "gp_train(gpm::GPModel; <optional keyword arguments>)\n\nFind Maximum Likelihood Estimate of Gaussian Process hyperparameters by maximising gp_loglikelihood, using Optim package. The optimisation target is gp_loglikelihood_log, with gradient computed by gp_loglikelihood_grad. Internally, this function optimises the MLE with respect to logarithms of hyperparameters. This is done for numerical stability. Logarithmisation and exponentiation is performed by this funtion, i.e. real hyperparameters, not logarithms, are taken in and returned back.\n\nBy default, Conjugate Gradient bounded box optimisation is used, as long as the gradient with respect to hyperparameters (covariance_grad) is implemented for the kernel function. If the gradient implementation is not provided, Nelder Mead optimiser is used by default.\n\nMandatory argument\n\ngpm: the GPModel, that contains the training data (x and y), the kernel and the starting hyperparameters that will be used for optimisation.\n\nOptional keyword arguments\n\noptimiser::Type{<:Optim.AbstractOptimizer}: the solver to use. If not given, then ConjugateGradient will be used for kernels that have gradient implementation, and NelderMead will be used for those that don\'t.\nhp_lower::AbstractArray{Float64, 1}: the lower boundary for box optimisation. Defaults to e^-10 for all hyperparameters.\nhp_upper::AbstractArray{Float64, 1}: the upper boundary for box optimisation. Defaults to e^10 for all hyperparameters.\nlog_level::Int: log level. Default is 0, which is no logging at all. 1 makes gp_train print basic information to standard output. 2 switches Optim logging on, in addition to 1.\n\nReturn\n\nThe list of all hyperparameters, including the standard deviation of the measurement noise sigma_n. Note that after this function returns, the hyperparameters of gpm will be set to the optimised value, and there is no need to call set_hyperparameters once again.\n\n\n\n\n\n"
},

{
    "location": "ref-gp/#Types-and-Functions-1",
    "page": "Gaussian Processes",
    "title": "Types and Functions",
    "category": "section",
    "text": "Modules = [GpABC]\nPages = [\"gp.jl\", \"gp_optimisation.jl\"]"
},

{
    "location": "ref-kernels/#",
    "page": "Kernels",
    "title": "Kernels",
    "category": "page",
    "text": ""
},

{
    "location": "ref-kernels/#Kernels-Reference-1",
    "page": "Kernels",
    "title": "Kernels Reference",
    "category": "section",
    "text": "GpABC functions and types for working with kernels."
},

{
    "location": "ref-kernels/#Index-1",
    "page": "Kernels",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"ref-kernels.md\"]"
},

{
    "location": "ref-kernels/#GpABC.AbstractGPKernel",
    "page": "Kernels",
    "title": "GpABC.AbstractGPKernel",
    "category": "type",
    "text": "AbstractGPKernel\n\nAbstract kernel type. User-defined kernels should derive from it.\n\nImplementations have to provide methods for get_hyperparameters_size and covariance. Methods for covariance_training, covariance_diagonal and covariance_grad are optional.\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.covariance-Tuple{AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.covariance",
    "category": "method",
    "text": "covariance(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nReturn the covariance matrix. Should be overridden by kernel implementations.\n\nArguments\n\nker: The kernel object. Implementations must override with their own subtype.\nlog_theta: natural logarithm of hyperparameters.\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\n\nReturn\n\nThe covariance matrix, of size n times m.\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.covariance_diagonal-Tuple{AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.covariance_diagonal",
    "category": "method",
    "text": "covariance_diagonal(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2})\n\nThis is a speedup version of covariance, which is invoked if the caller is not interested in the entire covariance matrix, but only needs the variance, i.e. the diagonal of the covariance matrix.\n\nDefault method just returns diag(covariance(...)), with x === z. Kernel implementations can optionally override it to achieve betrer performance, by not computing the non diagonal elements of covariance matrix.\n\nSee covariance for description of arguments.\n\nReturn\n\nThe 1-d array of variances, of size size(x, 1).\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.covariance_grad-Tuple{AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.covariance_grad",
    "category": "method",
    "text": "covariance_grad(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n\nReturn the gradient of the covariance function with respect to logarigthms of hyperparameters, based on the provided direction matrix.\n\nThis function can be optionally overridden by kernel implementations. If the gradient function is not provided, gp_train will fail back to NelderMead algorithm by default.\n\nArguments\n\nker: The kernel object. Implementations must override with their own subtype.\nlog_theta:  natural logarithm of hyperparameters\nx: Training data, reshaped into a 2-d array. x must have dimensions n times d.\nR the directional matrix, n times n\n\nR = frac1sigma_n^2(alpha * alpha^T - K^-1) alpha = K^-1y\n\nReturn\n\nA vector of size length(log_theta), whose j\'th element is equal to\n\ntr(R fracpartial Kpartial eta_j)\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.covariance_training-Tuple{AbstractGPKernel,AbstractArray{Float64,1},AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.covariance_training",
    "category": "method",
    "text": "covariance_training(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},\n    training_x::AbstractArray{Float64, 2})\n\nThis is a speedup version of covariance, which is only called during traing sequence. Intermediate matrices computed in this function for particular hyperparameters can be cached and reused subsequently, either in this function or in covariance_grad\n\nDefault method just delegates to covariance with x === z. Kernel implementations can optionally override it for betrer performance.\n\nSee covariance for description of arguments and return values.\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.get_hyperparameters_size-Tuple{AbstractGPKernel,AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.get_hyperparameters_size",
    "category": "method",
    "text": "get_hyperparameters_size(kernel::AbstractGPKernel, training_data::AbstractArray{Float64, 2})\n\nReturn the number of hyperparameters for used by this kernel on this training data set. Should be overridden by kernel implementations.\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.MaternArdKernel",
    "page": "Kernels",
    "title": "GpABC.MaternArdKernel",
    "category": "type",
    "text": "MaternArdKernel <: AbstractGPKernel\n\nMatérn kernel with distinct length scale for each dimention, l_k. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.MaternIsoKernel",
    "page": "Kernels",
    "title": "GpABC.MaternIsoKernel",
    "category": "type",
    "text": "MaternIsoKernel <: AbstractGPKernel\n\nMatérn kernel with uniform length scale across all dimensions, l. Parameter nu (nu) is passed in constructor. Currently, only values of nu=1, nu=3 and nu=5 are supported.\n\nbeginaligned\nK_nu=1(r) = sigma_f^2e^-sqrtr\nK_nu=3(r) = sigma_f^2(1 + sqrt3r)e^-sqrt3r\nK_nu=5(r) = sigma_f^2(1 + sqrt3r + frac53r)e^-sqrt5r\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.SquaredExponentialArdKernel",
    "page": "Kernels",
    "title": "GpABC.SquaredExponentialArdKernel",
    "category": "type",
    "text": "SquaredExponentialArdKernel <: AbstractGPKernel\n\nSquared exponential kernel with distinct length scale for each dimention, l_k.\n\nbeginaligned\nK(r)  = sigma_f^2 e^-r2 \nr_ij  = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nThe length of hyperparameters array for this kernel depends on the dimensionality of the data. Assuming each data point is a vector in a d-dimensional space, this kernel needs d+1 hyperparameters, in the following order:\n\nsigma_f: the signal standard deviation\nl_1 ldots l_d: the length scales for each dimension\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.SquaredExponentialIsoKernel",
    "page": "Kernels",
    "title": "GpABC.SquaredExponentialIsoKernel",
    "category": "type",
    "text": "SquaredExponentialIsoKernel <: AbstractGPKernel\n\nSquared exponential kernel with uniform length scale across all dimensions, l.\n\nbeginaligned\nK(r)  = sigma_f^2 e^-r2 \nr_ij  = sum_k=1^dfrac(x_ik-z_jk)^2l^2\nendaligned\n\nr_ij are computed by scaled_squared_distance\n\nHyperparameters\n\nHyperparameters vector for this kernel must contain two elements, in the following order:\n\nsigma_f: the signal standard deviation\nl: the length scale\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.ExponentialArdKernel-Tuple{}",
    "page": "Kernels",
    "title": "GpABC.ExponentialArdKernel",
    "category": "method",
    "text": "ExponentialArdKernel\n\nAlias for MaternArdKernel(1)\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.ExponentialIsoKernel-Tuple{}",
    "page": "Kernels",
    "title": "GpABC.ExponentialIsoKernel",
    "category": "method",
    "text": "ExponentialIsoKernel\n\nAlias for MaternIsoKernel(1)\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.scaled_squared_distance-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.scaled_squared_distance",
    "category": "method",
    "text": "scaled_squared_distance(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})\n\nCompute the scaled squared distance between x and z:\n\nr_ij = sum_k=1^dfrac(x_ik-z_jk)^2l_k^2\n\nThe gradient of this function with respect to length scale hyperparameter(s) is returned by scaled_squared_distance_grad.\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\n\nReturn\n\nAn n times m matrix of scaled squared distances\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#GpABC.scaled_squared_distance_grad-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2},AbstractArray{Float64,2},AbstractArray{Float64,2}}",
    "page": "Kernels",
    "title": "GpABC.scaled_squared_distance_grad",
    "category": "method",
    "text": "scaled_squared_distance_grad(log_ell::AbstractArray{Float64, 1},\n    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})\n\nReturn the gradient of the scaled_squared_distance function with respect to logarigthms of length scales, based on the provided direction matrix.\n\nArguments\n\nx, z: Input data, reshaped into 2-d arrays. x must have dimensions n times d; z must have dimensions m times d.\nlog_ell: logarithm of length scale(s). Can either be an array of size one (isotropic), or an array of size d (ARD)\nR the direction matrix, n times m. This can be used to compute the gradient of a function that depends on scaled_squared_distance via the chain rule.\n\nReturn\n\nA vector of size length(log_ell), whose k\'th element is equal to\n\ntexttr(R fracpartial Kpartial l_k)\n\n\n\n\n\n"
},

{
    "location": "ref-kernels/#Types-and-Functions-1",
    "page": "Kernels",
    "title": "Types and Functions",
    "category": "section",
    "text": "Modules = [GpABC]\nPages = [\"_kernel.jl\", \"_kernels.jl\", \"scaled_squared_distance.jl\"]"
},

{
    "location": "faq/#",
    "page": "FAQ",
    "title": "FAQ",
    "category": "page",
    "text": ""
},

{
    "location": "faq/#Frequently-Asked-Questions-1",
    "page": "FAQ",
    "title": "Frequently Asked Questions",
    "category": "section",
    "text": ""
},

{
    "location": "faq/#Q:-My-computation-terminates-before-sampling-sufficient-number-of-particles.-How-do-I-make-it-run-longer?-1",
    "page": "FAQ",
    "title": "Q: My computation terminates before sampling sufficient number of particles. How do I make it run longer?",
    "category": "section",
    "text": "A: All parameter inference functions accept an optional parameter max_iter, which specifies the number of iterations to run. See ABC Reference for more details."
},

{
    "location": "faq/#Q:-When-running-the-example-on-my-machine,-emulation-provides-no-performance-benefit-or-even-takes-longer-than-emulation.-How-do-I-speed-up-emulation?-1",
    "page": "FAQ",
    "title": "Q: When running the example on my machine, emulation provides no performance benefit or even takes longer than emulation. How do I speed up emulation?",
    "category": "section",
    "text": "A: The best way to reduce the time it takes to run the emulation is to decrease the batch_size parameter. Please refer to docs for EmulatedABCRejection end EmulatedABCSMC. Machines with larger amount of RAM would be able to handle larger batches without performance degradation."
},

{
    "location": "faq/#Q:-How-do-I-generate-a-plot-that-is-similar-to-the-one-in-the-paper?-1",
    "page": "FAQ",
    "title": "Q: How do I generate a plot that is similar to the one in the paper?",
    "category": "section",
    "text": "A: You will need to run ABC-SMC parameter estimation using both emulation and simulation for the same problem. Then, assuming you have Plots package installed, just runjulia> plot(emu_out, sim_out, true_params)Use pyplot() backend for best results."
},

{
    "location": "faq/#Q:-How-do-I-run-the-example-notebooks-locally?-1",
    "page": "FAQ",
    "title": "Q: How do I run the example notebooks locally?",
    "category": "section",
    "text": "A:Make sure that Jupyter is installed, along with Julia and its Jupyter Kernel.\nClone or download GpABC.jl to your machine.\nRun jupyter from the examples directory of the local copy of GpABC.jl:$ cd GpABC.jl/examples\n$ jupyter notebookThe first line of the first cell, that contains something likeimport Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()\nusing ...will download all the dependencies.All notebooks were tested under Julia 1.2"
},

]}
