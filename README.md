# GpABC.jl

[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tanhevg.github.io/GpABC.jl/latest)
[![Build Status](https://travis-ci.org/tanhevg/GpABC.jl.svg?branch=master)](https://travis-ci.org/tanhevg/GpABC.jl)
[![codecov.io](http://codecov.io/github/tanhevg/GpABC.jl/coverage.svg?branch=master)](http://codecov.io/github/tanhevg/GpABC.jl?branch=master)


This Julia package allows Approximate Bayesian Computation for ordinary and stochastic differential equation models. Approximate Bayesian Computation enables:
* Likelihood-free parameter estimation
* Likelihood-free model selection

`GpABC` offers this functionality using two approaches:

* Simulation - full simulations of the differential equation model(s) for different parameter vectors.
* Emulation - a small number of simulations can be used to train a regression model (the _emulator_), which approximates full model simulations.

`GpABC` includes Gaussian Process regression as an emulator, but custom emulators can also be used. The Gaussian Process implementation can also be used "standalone" for any regression task.

<!-- Two types of ABC are available:

* Rejection Approximate Bayesian Computation
* Approximate Bayesian Computation-Sequential Monte Carlo (ABC-SMC)

 with Gaussian
Process Regression (GPR). This emulation technique is then applied to parameter inference
using Approximate Bayesian Computation based on Sequential Monte Carlo (ABC-SMC).
Emulating the model on each step of ABC-SMC, instead of simulating it, brings in significant performance benefits.
Theoretical background for this work is covered in the papers and textbooks listed in [References](References) section. -->

## Dependencies
`GpABC` depends on [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) [**TODO reference** ] for training Gaussian Process hyperparameters. Utility functions provided by `GpABC` also make use of [`Distributions`](https://github.com/JuliaStats/Distributions.jl). [`DifferentialEquations`](https://github.com/JuliaDiffEq/DifferentialEquations.jl) is used throughout the examples for simulating ODEs and SDEs, but there is no direct dependency on it in the package.

## Installation

**TODO** add `GpABC` to Julia package manager

```julia
Pkg.add("GpABC")
```

## Use cases
Although the primary objective of this package is parameter estimation of ODEs and SDEs
with ABC-SMC, using GPR emulation, each of the intermediate steps can be run independently:
* Run Gaussian Process regression ([Example 1](https://tanhevg.github.io/GpABC.jl/latest/examples/#example-2), [Example 2](https://tanhevg.github.io/GpABC.jl/latest/examples/#example-4))
* Estimate model parameters using Rejection ABC without emulation (simulation only) [ **TODO link to example** ]
* Estimate model parameters using Rejection ABC with GPR emulation [ **TODO link to example** ]
* Estimate model parameters using Sequential Monte Carlo (SMC) ABC without emulation  (simulation only) [ **TODO link to example** ]
* Estimate model parameters using ABC-SMC with GPR emulation [ **TODO link to example** ]

## References
**TODO**
