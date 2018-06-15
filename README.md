# GpAbc.jl

[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tanhevg.github.io/GpAbc.jl/latest)
[![Build Status](https://travis-ci.org/tanhevg/GpAbc.jl.svg?branch=master)](https://travis-ci.org/tanhevg/GpAbc.jl)
[![codecov.io](http://codecov.io/github/tanhevg/GpAbc.jl/coverage.svg?branch=master)](http://codecov.io/github/tanhevg/GpAbc.jl?branch=master)


This Julia package allows emulating ordinary and stochastic differential equations (ODEs and SDEs) with Gaussian
Process Regression (GPR). This emulation technique is then applied to parameter inference
using Approximate Bayesian Computation based on Sequential Monte Carlo (ABC-SMC).
Emulating the model on each step of ABC-SMC, instead of simulating it, brings in significant performance benefits.
Theoretical background for this work is covered in the papers and textbooks listed in [References](References) section.

## Dependencies
`GpAbc` depends on [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) [**TODO reference** ] for training Gaussian Process hyperparameters. Utility functions provided by `GpAbc` also make use of [`Distributions`](https://github.com/JuliaStats/Distributions.jl). [`DifferentialEquations`](https://github.com/JuliaDiffEq/DifferentialEquations.jl) is used throughout the examples for simulating ODEs and SDEs, but there is no direct dependency on it in the package.

## Installation

**TODO** add the `JpAbc` to Julia package manager

```julia
Pkg.add("GpAbc")
```

## Use cases
Although the primary objective of this package is parameter estimation of ODEs and SDEs
with ABC-SMC, using GPR emulation, each of the intermediate steps can be run independently:
* Run Gaussian Process regression ([Example 1](https://tanhevg.github.io/GpAbc.jl/latest/examples/#readme-example-1), [Example 2](https://tanhevg.github.io/GpAbc.jl/latest/examples/#readme-example-2))
* Estimate model parameters using Rejection ABC without emulation (simulation only) [ **TODO link to example** ]
* Estimate model parameters using Rejection ABC with GPR emulation [ **TODO link to example** ]
* Estimate model parameters using Sequential Monte Carlo (SMC) ABC without emulation  (simulation only) [ **TODO link to example** ]
* Estimate model parameters using ABC-SMC with GPR emulation [ **TODO link to example** ]

## References
**TODO**
