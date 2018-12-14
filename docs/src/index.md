# GpABC

`GpABC` provides algorithms for likelihood - free [parameter inference](@ref abc-overview) and [model selection](@ref ms-overview) using Approximate Bayesian Computation (**ABC**). Two sets of algorithms are available:

* Simulation based - full simulations of the model(s) is done on each step of ABC.
* Emulation based - a small number of simulations can be used to train a regression model (the *emulator*), which is then used to approximate model simulation results during ABC.

`GpABC` offers [Gaussian Process Regression](@ref gp-overview) (**GPR**) as an emulator, but custom emulators can also be used. GPR can also be used standalone, for any regression task.

Stochastic models, that don't conform to Gaussian Process Prior assumption, are supported via [Linear Noise Approximation](@ref lna-overview) (**LNA**).

## Examples
- [ABC parameter estimation example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/abc-example.ipynb)
- [ABC model selection example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/model-selection-example.ipynb)
- [LNA example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/lna-example.ipynb)
- [Gaussian Process regression example](https://github.com/tanhevg/GpABC.jl/blob/master/examples/gp-example.ipynb)

## Dependencies
-  [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) - for training Gaussian Process hyperparameters.
- [`Distributions`](https://github.com/JuliaStats/Distributions.jl) - probability distributions.
- [`Distances`](https://github.com/JuliaStats/Distances.jl) - distance functions
- [`DifferentialEquations`](https://github.com/JuliaDiffEq/DifferentialEquations.jl) - for solving ODEs for LNA, and also used throughout the examples for model simulation (ODEs and SDEs)
- [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) - automatic differentiation is also used by LNA

## References
**TODO** papers
