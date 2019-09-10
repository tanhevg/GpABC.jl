# GpABC

`GpABC` provides algorithms for likelihood - free [parameter inference](@ref abc-overview) and [model selection](@ref ms-overview) using Approximate Bayesian Computation (**ABC**). Two sets of algorithms are available:

* Simulation based - full simulations of the model(s) is done on each step of ABC.
* Emulation based - a small number of simulations can be used to train a regression model (the *emulator*), which is then used to approximate model simulation results during ABC.

`GpABC` offers [Gaussian Process Regression](@ref gp-overview) (**GPR**) as an emulator, but custom emulators can also be used. GPR can also be used standalone, for any regression task.

Stochastic models, that don't conform to Gaussian Process Prior assumption, are supported via [Linear Noise Approximation](@ref lna-overview) (**LNA**).


## Installation

`GpABC` can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add https://github.com/tanhevg/GpABC.jl
```

## Notation

In parts of this manual that deal with Gaussian Processes and kernels,
we denote the number of training points as $n$, and the number of
test points as $m$. The number of dimensions is denoted as $d$.

In the context of ABC, vectors in parameter space (``\theta``) are referred to as _particles_.
Particles that are used for training the emulator (`training_x`) are called _design points_.
To generate the distances for training the emulator (`training_y`), the model must be simulated for the design points.


## Examples
- [ABC parameter estimation example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/abc-example.ipynb)
- [ABC model selection example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/model-selection-example.ipynb)
- [Stochastic Inference (LNA) example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/lna-example.ipynb)
- [Gaussian Process regression example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/gp-example.ipynb)

## Dependencies
-  [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) - for training Gaussian Process hyperparameters.
- [`Distributions`](https://github.com/JuliaStats/Distributions.jl) - probability distributions.
- [`Distances`](https://github.com/JuliaStats/Distances.jl) - distance functions
- [`DifferentialEquations`](https://github.com/JuliaDiffEq/DifferentialEquations.jl) - for solving ODEs for LNA, and also used throughout the examples for model simulation (ODEs and SDEs)
- [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) - automatic differentiation is also used by LNA

## References

- Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. *Interface*, (July 2008), 187–202. [https://doi.org/10.1098/rsif.2008.0172](https://doi.org/10.1098/rsif.2008.0172)
- Filippi, S., Barnes, C. P., Cornebise, J., & Stumpf, M. P. H. (2013). On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. *Statistical Applications in Genetics and Molecular Biology*, 12(1), 87–107. [https://doi.org/10.1515/sagmb-2012-0069](https://doi.org/10.1515/sagmb-2012-0069)
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. ISBN 0-262-18253-X. [http://www.gaussianprocess.org/gpml](http://www.gaussianprocess.org/gpml)
- Schnoerr, D., Sanguinetti, G., & Grima, R. (2017). Approximation and inference methods for stochastic biochemical kinetics—a tutorial review. *Journal of Physics A: Mathematical and Theoretical*, 50(9), 093001. [https://doi.org/10.1088/1751-8121/aa54d9](https://doi.org/10.1088/1751-8121/aa54d9)
- Karlebach, G., & Shamir, R. (2008). Modelling and analysis of gene regulatory networks. *Nature Reviews Molecular Cell Biology*, 9(10), 770–780. [https://doi.org/10.1038/nrm2503](https://doi.org/10.1038/nrm2503)
