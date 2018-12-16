# [Gaussian Processes Regression Overview](@id gp-overview)

**TODO** paper1

A *Gaussian Process* (**GP**) is a collection of random variables, any finite number of which have a joint Gaussian distribution. This assumption is often referred to as GP prior. In a regression setting, we are going to use GPs to approximate an unknown function ``f(x)``. We denote the training data set as ``\mathbf{x}`` and the test data set as ``\mathbf{x^*}``. Function values on training and test data sets are denoted as ``\mathbf{y} = \mathbf{f(x)}``, and ``\mathbf{y^*} = \mathbf{f(x^*)}``, respectively (in vectorised form). In this notation, assuming the zero mean, the GP prior can be written as

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

Here, ``K`` is the covariance matrix computed on the training data, ``K^{**}`` is the covariance matrix computed on the test data, and ``K^{*}`` is the covariance matrix between the training and test data. We assume that the covariance between any two points ``x`` and ``x^{'}`` is given by a *kernel function* ``k(x, x^{'})``, or in matrix notation, ``K_{ij} = k(x_i, x_j)`` 
