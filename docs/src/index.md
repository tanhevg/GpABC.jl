
* [Notation](@ref)
* [Basic Usage](@ref)
* [Training the GP](@ref)
* [Kernels](@ref)

**TODO markdown does not support include, so copy-paste content from /README.md once it is finalised**

## Notation

Throughout this manual, we denote the number of training points as $n$, and the number of
test points as $m$. The number of dimensions is denoted as $d$. For one-dimensional case, where each individual training and test point
is just a real number, both one-dimensional and two-dimensional arrays are accepted
as inputs. In [Basic Gaussian Process Regression Example](@ref example-1) `training_x` can either be a vector of size $n$, or an $n \times 1$
matrix. For a multidimensional case, where test and training points are elements of a
$d$-dimentional space, all inputs have to be row major, so `training_x` and `test_x` become
an $n \times d$ and an $m \times d$ matrices, respectively.

## Basic Usage

The package is built around a type [`GPModel`](@ref), which encapsulates all the information
required for training the Gaussian Process and performing the regression. In the simplest
scenario the user would instantiate this type with some training data and labels, provide
the hyperparameters and run the regression. By default, [`SquaredExponentialIsoKernel`](@ref) will be used. This scenario is illustrated by [Basic Gaussian Process Regression Example](@ref example-1).

## Training the GP

Normally, kernel hyperparameters are not known in advance. In this scenario the training function [`gp_train`](@ref) should be used to find the Maximum Likelihood Estimate (MLE) of hyperparameters. This is demonstrated in [Optimising Hyperparameters for GP Regression Example](@ref example-2).

GaussProABC uses [Optim](https://github.com/JuliaNLSolvers/Optim.jl) package for optimising the hyperparameters. By default,
[Conjugate Gradient](http://julianlsolvers.github.io/Optim.jl/stable/algo/cg/) bounded box optimisation is used, as long as the gradient
with respect to hyperparameters is implemented for the kernel function. If the gradient
implementation is not provided, [Nelder Mead](http://julianlsolvers.github.io/Optim.jl/stable/algo/nelder_mead/) optimiser is used by default.

The starting point of the optimisation can be specified by calling [`set_hyperparameters`](@ref).
If the starting point has not been provided, optimisation will start from all hyperparameters
set to 1. Default upper and lower bounds are set to $e^{10}$ and $e^{−10}$ , respectively, for each
hyperparameter.

For numerical stability the package uses logarithms of hyperparameters internally, when
calling the log likelihood and kernel functions. Logarithmisation and exponentiation
back takes place in [`gp_train`](@ref) function.

The log likelihood function with log hyperparameters is implemented
by [`gp_loglikelihood_log`](@ref). This is the target function of the optimisation procedure in
[`gp_train`](@ref). There is also a version of log likelihood with actual (non-log) hyperparameters: [`gp_loglikelihood`](@ref). The gradient of the log likelihood function with
respect to logged hyperparameters is implemented by [`gp_loglikelihood_grad`](@ref).

Depending on the kernel, it is not uncommon for the log likelihood function to have
multiple local optima. If a trained GP produces an unsatisfactory data fit, one
possible workaround is trying to run [`gp_train`](@ref) several times with random starting points. This approach is demonstrated in [Advanced Usage of gp_train example](@ref example-3).

`Optim` has a built in constraint of running no more than 1000 iterations of any optimisation
algorithm. `GpABC` relies on this feature to ensure that the training procedure
does not get stuck forever. As a consequence, the optimizer might exit prematurely,
before reaching the local optimum. Setting `log_level` argument of [`gp_train`](@ref) to a value
greater than zero will make it log its actions to standard output, including
whether the local minimum has been reached or not.

## Kernels

`GpABC` ships with an extensible library of kernel functions. Each kernel is represented with a type that derives from [`AbstractGPKernel`](@ref):
* [`SquaredExponentialIsoKernel`](@ref)
* [`SquaredExponentialArdKernel`](@ref)
* [`MaternIsoKernel`](@ref)
* [`MaternArdKernel`](@ref)
* [`ExponentialIsoKernel`](@ref)
* [`ExponentialArdKernel`](@ref)

These kernels rely on matrix of scaled squared distances between training/test inputs ``[r_{ij}]``, which is computed by [`scaled_squared_distance`](@ref) function. The gradient vector of scaled squared distance derivatives with respect to
length scale hyperparameter(s) is returned by [`scaled_squared_distance_grad`](@ref) function.


The kernel covariance matrix is returned by function [`covariance`](@ref). Optional speedups of this
function [`covariance_diagonal`](@ref) and [`covariance_training`](@ref) are implemented for the
pre-shipped kernels.
The gradient with respect to log hyperparameters is computed by [`covariance_grad`](@ref). The `log_theta` argument refers to the logarithms of kernel hyperparameters.
Note that hyperparameters that do not affect the kernel (e.g. ``\sigma_n`` ) are not included in
`log_theta`.

Custom kernels functions can be implemented  by adding more types that inherit from [`AbstractGPKernel`](@ref). This is demonstrated in [Using a Custom Kernel Example](@ref example-4)
