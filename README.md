# GpABC.jl

[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tanhevg.github.io/GpABC.jl/latest)
[![Build Status](https://travis-ci.org/tanhevg/GpABC.jl.svg?branch=master)](https://travis-ci.org/tanhevg/GpABC.jl)
[![codecov.io](http://codecov.io/github/tanhevg/GpABC.jl/coverage.svg?branch=master)](http://codecov.io/github/tanhevg/GpABC.jl?branch=master)
[![DOI:10.1093/bioinformatics/btaa078](https://zenodo.org/badge/DOI/10.1093/bioinformatics/btaa078.svg)](https://doi.org/10.1093/bioinformatics/btaa078)



`GpABC` provides algorithms for likelihood - free [parameter inference](https://tanhevg.github.io/GpABC.jl/latest/overview-abc/) and [model selection](https://tanhevg.github.io/GpABC.jl/latest/overview-ms/) using Approximate Bayesian Computation (**ABC**). Two sets of algorithms are available:

* Simulation based - full simulations of the model(s) is done on each step of ABC.
* Emulation based - a small number of simulations can be used to train a regression model (the *emulator*), which is then used to approximate model simulation results during ABC.

`GpABC` offers [Gaussian Process Regression](https://tanhevg.github.io/GpABC.jl/latest/overview-gp/) (**GPR**) as an emulator, but custom emulators can also be used. GPR can also be used standalone, for any regression task.

Stochastic models, that don't conform to Gaussian Process Prior assumption, are supported via [Linear Noise Approximation](https://tanhevg.github.io/GpABC.jl/latest/overview-lna/) (**LNA**).


## Installation

`GpABC` can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add https://github.com/tanhevg/GpABC.jl
```
## Paper

If you are using GpABC in research, please cite our paper:

**GpABC: a Julia package for approximate Bayesian computation with Gaussian process emulation**

Evgeny Tankhilevich, Jonathan Ish-Horowicz, Tara Hameed, Elisabeth Roesch, Istvan Kleijn, Michael PH Stumpf, Fei He

https://www.biorxiv.org/content/10.1101/769299v1

doi: [10.1093/bioinformatics/btaa078](https://doi.org/10.1093/bioinformatics/btaa078)

```
@article{10.1093/bioinformatics/btaa078,
    author = {Tankhilevich, Evgeny and Ish-Horowicz, Jonathan and Hameed, Tara and Roesch, Elisabeth and Kleijn, Istvan and Stumpf, Michael P H and He, Fei},
    title = "{GpABC: a Julia package for approximate Bayesian computation with Gaussian process emulation}",
    journal = {Bioinformatics},
    year = {2020},
    month = {02},
    abstract = "{Approximate Bayesian computation (ABC) is an important framework within which to infer the structure and parameters of a systems biology model. It is especially suitable for biological systems with stochastic and nonlinear dynamics, for which the likelihood functions are intractable. However, the associated computational cost often limits ABC to models that are relatively quick to simulate in practice.We here present a Julia package, GpABC, that implements parameter inference and model selection for deterministic or stochastic models using i) standard rejection ABC or ABC-SMC, or ii) ABC with Gaussian process emulation. The latter significantly reduces the computational cost.https://github.com/tanhevg/GpABC.jlSupplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa078},
    url = {https://doi.org/10.1093/bioinformatics/btaa078},
    note = {btaa078},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaa078/32353462/btaa078.pdf},
}
```

## Examples
### ABC Parameter Inference
[![Github](https://img.shields.io/badge/view-github-lightgrey?logo=github)](examples/abc-example.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tanhevg/GpABC.jl/master?filepath=examples%2Fabc-example.ipynb) [![NBViewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/tanhevg/GpABC.jl/blob/master/examples/abc-example.ipynb)
### Gaussian Process Regression
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tanhevg/GpABC.jl/master?filepath=examples%2Fgp-example.ipynb) [![NBViewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/tanhevg/GpABC.jl/blob/master/examples/gp-example.ipynb)
### Stochastic Inference (LNA)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tanhevg/GpABC.jl/master?filepath=examples%2Flna-example.ipynb) [![NBViewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/tanhevg/GpABC.jl/blob/master/examples/lna-example.ipynb)
### Model Selection
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tanhevg/GpABC.jl/master?filepath=examples%2Fmodel-selection-example.ipynb) [![NBViewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/tanhevg/GpABC.jl/blob/master/examples/model-selection-example.ipynb)

## Dependencies
-  [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) - for training Gaussian Process hyperparameters.
- [`Distributions`](https://github.com/JuliaStats/Distributions.jl) - probability distributions.
- [`Distances`](https://github.com/JuliaStats/Distances.jl) - distance functions
- [`OrdinaryDiffEq`](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) - for solving ODEs for LNA, and also used throughout the examples for model simulation (ODEs and SDEs)
- [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) - automatic differentiation is also used by LNA
- `PlotUtils`, `RecipesBase`, `Colors`, `KernelDensity` - for plotting figures

## References

- Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. *Interface*, (July 2008), 187–202. https://doi.org/10.1098/rsif.2008.0172
- Filippi, S., Barnes, C. P., Cornebise, J., & Stumpf, M. P. H. (2013). On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. *Statistical Applications in Genetics and Molecular Biology*, 12(1), 87–107. https://doi.org/10.1515/sagmb-2012-0069
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. ISBN 0-262-18253-X. http://www.gaussianprocess.org/gpml
- Schnoerr, D., Sanguinetti, G., & Grima, R. (2017). Approximation and inference methods for stochastic biochemical kinetics—a tutorial review. *Journal of Physics A: Mathematical and Theoretical*, 50(9), 093001. https://doi.org/10.1088/1751-8121/aa54d9
- Karlebach, G., & Shamir, R. (2008). Modelling and analysis of gene regulatory networks. *Nature Reviews Molecular Cell Biology*, 9(10), 770–780. https://doi.org/10.1038/nrm2503
