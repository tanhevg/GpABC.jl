# [Gaussian Processes Regression Overview](@id gp-overview)

An excellent introduction to Gaussian Processes is available in the book by (Rasmussen and Williams, 2006).

```@contents
Pages = ["overview-gp.md"]
```

## Gaussian Process, Prior and Posterior

A *Gaussian Process* (**GP**) is a collection of random variables, any finite number of which have a joint Gaussian distribution. This assumption is often referred to as the *GP prior*. In a regression setting, we are going to use GPs to approximate an unknown function ``f(x)``, ``x`` being a ``d``-dimensional feature vector, ``x \in \mathbb{R}^d``. We assume that our training data set contains of ``n`` points in `` \mathbb{R}^d``, and the test set - of ``m`` points in `` \mathbb{R}^d``. We denote the training data set as ``\mathbf{x}, \mathbf{x} \in \mathbb{R}^{n \times d}`` and the test data set as ``\mathbf{x^*}, \mathbf{x^*} \in \mathbb{R}^{m \times d}``. Function values on training and test data sets are denoted as ``\mathbf{y} = \mathbf{f(x)}``, and ``\mathbf{y^*} = \mathbf{f(x^*)}``, respectively (in vectorised form). We also assume that the mean of the prior Gaussian distribution is zero, and its covariance matrix is known. Furthermore, we split the covariance matrix into the following regions:

- ``K``: the covariance matrix computed on the training data, ``K \in \mathbb{R}^{n \times n}``
- ``K^{**}``: the covariance matrix computed on the test data, ``K^{**} \in \mathbb{R}^{m \times m}``
- ``K^{*}``: the covariance matrix between the training and test data, ``K^{*} \in \mathbb{R}^{n \times m}``

In this notation, the GP prior can be written as

```math
\left[ \begin{matrix}
\mathbf{y}\\
\mathbf{y^*}
\end{matrix} \right]
\sim \mathcal{N} \left( 0,
\left[ \begin{matrix}
K & K^*\\
K^{*\top} & K^{**}
\end{matrix} \right] \right)
```

The desired approximation of ``f`` in ``\mathbf{x^*}`` is the conditional distribution of ``\mathbf{y^*}``, given ``\mathbf{x}``, ``\mathbf{y}`` and ``\mathbf{x^*}``. This distrubution, referred to as *GP posterior*, can be derived from the GP prior and the properties of a multivariate Normal distribution:

```math
\begin{align*}
\mathbf{y^* | x, y, x^*} & \sim \mathcal{N}(\mathbf{\tilde{y}}, \tilde{K}) \\
\mathbf{\tilde{y}} & = K^{*\top} K^{-1} \mathbf{y} \\
\tilde{K} & = K^{**} - K^{*\top} K^{-1} K^*
\end{align*}
```

``\mathbf{\tilde{y}}`` and ``\mathbf{\tilde{K}}`` are, respectively, the mean vector and the covariance matrix of the GP posterior. Often, we are not interested in non-diagonal elements of ``\mathbf{\tilde{K}}``. In such cases just the vector of diagonal elements is reported.

## Kernels and Hyperparameters

We assume that the covariance between any two points ``x`` and ``x^{'}`` is given by a *kernel function* ``k(x, x^{'})``, or in matrix notation, ``K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)``. This kernel function is parameterised by a vector of *hyperparameters* ``\mathbf{\eta} = \eta_1, \ldots, \eta_p``. The covariance matrix is thus also dependent on ``\mathbf{\eta}``: ``K = K(\mathbf{\eta})``.

The optimal values of hyperparameters ``\hat{\eta}`` can be obtained by finding the maximum value of log likelihood of the GP prior:

```math
\begin{align*}
\log p(\mathbf{y|\eta}) &= -\frac{1}{2}\mathbf{y}^\top K^{-1} \mathbf{y} - \frac{1}{2}|K| - \frac{n}{2}\log(2\pi) \\
\hat{\eta} &= \underset{\mathbf{\eta}}{\text{argmax}}(\log p(\mathbf{y}|\mathbf{\eta}))
\end{align*}
```

In `GpABC` this optimisation is performed using [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) package.
By default,
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
possible workaround is trying to run [`gp_train`](@ref) several times with random starting points.

`Optim` has a built in constraint of running no more than 1000 iterations of any optimisation
algorithm. `GpABC` relies on this feature to ensure that the training procedure
does not get stuck forever. As a consequence, the optimizer might exit prematurely,
before reaching the local optimum. Setting `log_level` argument of [`gp_train`](@ref) to a value
greater than zero will make it log its actions to standard output, including
whether the local minimum has been reached or not.

It is often convenient to model the measurement noise in the training data separately. This amounts to a normally distributed random variable being added to ``\mathbf{y}``. Denoting the variance of this random noise as ``\sigma_n``, this is equivalent to altering the covariance matrix to ``K_y = K + \sigma_n I``, where ``I`` is the identity matrix. Noise variance ``\sigma_n`` is also a hyperparameter, that must be optimised with the rest of kernel hyperparameters. `GpABC` uses a joint hyperparameter vector, where ``\sigma_n`` is always the last element.

## Custom Kernels

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

Custom kernels functions can be implemented  by adding more types that inherit from [`AbstractGPKernel`](@ref).

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. ISBN 0-262-18253-X. [http://www.gaussianprocess.org/gpml](http://www.gaussianprocess.org/gpml)
