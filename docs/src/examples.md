# Examples

* [Basic Gaussian Process Regression](@ref example-1)
* [Optimising Hyperparameters for GP Regression](@ref example-2)
* [Advanced Usage of gp_train](@ref example-3)
* [Using a Custom Kernel](@ref example-4)

### [Basic Gaussian Process Regression](@id example-1)

```julia
using GpABC, Distributions, PyPlot

# prepare the data
n = 30
f(x) = x.ˆ2 + 10 * sin.(x) # the latent function

training_x = sort(rand(Uniform(-10, 10), n))
training_y = f(training_x)
training_y += 20 * (rand(n) - 0.5) # add some noise
test_x = collect(linspace(min(training_x...), max(training_x...), 1000))

 # SquaredExponentialIsoKernel is used by default
gpm = GPModel(training_x, training_y)

# pretend we know the hyperparameters in advance
# σ_f = 37.08; l = 1.0; σ_n = 6.58. See SquaredExponentialIsoKernel documentation for details
set_hyperparameters(gpm, [37.08, 1.0, 6.58])
(test_y, test_var) = gp_regression(test_x, gpm)

plot(test_x, [test_y f(test)]) # ... and more sophisticated plotting
```

### [Optimising Hyperparameters for GP Regression](@id example-2)

Based on [Basic Gaussian Process Regression](@ref example-1), but with added optimisation
of hyperparameters:

```julia
using GpABC

# prepare the data ...

gpm = GPModel(training_x, training_y)

 # by default, the optimiser will start with all hyperparameters set to 1,
 # constrained between exp(-10) and exp(10)
theta_mle = gp_train(gpm)

# optimised hyperparameters are stored in gpm, so no need to pass them again
(test_y, test_var) = gp_regression(test_x, gpm)
```

### [Advanced Usage of gp_train](@id example-3)

```julia
using GpABC, Optim, Distributions

function gp_train_advanced(gpm::GPModel, attempts::Int)
    # Initialise the bounds, with special treatment for the second hyperparameter
    p = get_hyperparameters_size(gpm)
    bound_template = ones(p)
    upper_bound = bound_template * 10
    upper_bound[2] = 2
    lower_bound = bound_template * -10
    lower_bound[2] = -1

    # Starting point will be sampled from a Multivariate Uniform distribution
    start_point_distr = MvUniform(lower_bound, upper_bound)

    # Run several attempts of training and store the
    # minimiser hyperparameters and the value of log likelihood function
    hypers = Array{Float64}(attempts, p)
    likelihood_values = Array{Float64}(attempts)
    for i=1:attempts
        set_hyperparameters(gpm, exp.(rand(start_point_distr)))
        hypers[i, :] = gp_train(gpm,
            optimisation_solver_type=SimulatedAnnealing, # note the solver type
            hp_lower=lower_bound, hp_upper=upper_bound, log_level=1)
        likelihood_values[i] = gp_loglikelihood(gpm)
    end
    # Retain the hyperparameters where the maximum log likelihood function is attained
    gpm.gp_hyperparameters = hypers[indmax(likelihood_values), :]
end
```

### [Using a Custom Kernel](@id example-4)

The methods below should be implemented for the custom kernel, unless indicated as optional.
Please see reference documentation for detailed description of each method and parameter.

```julia
using GpABC
import GpABC.covariance, GpABC.get_hyperparameters_size, GpABC.covariance_diagonal,
    GpABC.covariance_training, GpABC.covariance_grad

"""
   This is the new kernel that we are adding
"""
type MyCustomkernel <: AbstractGPKernel

    # optional cache of matrices that could be re-used between calls to
    # covariance_training and covariance_grad, keyed by hyperparameters
    cache::MyCustomCache
end

"""
    Report the number of hyperparameters required by the new kernel
"""
function get_hyperparameters_size(ker::MyCustomkernel, training_data::AbstractArray{Float64, 2})
    # ...
end

"""
    Covariance function of the new kernel.

    Return the covariance matrix. Assuming x is an n by d matrix, and z is an m by d matrix,
    this should return an n by m matrix. Use `scaled_squared_distance` helper function here.
"""
function covariance(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},
    x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})
    # ...
end

"""
    Optional speedup of `covariance` function, that is invoked when the calling code is
    only interested in variance (i.e. diagonal elements of the covariance) of the kernel.
"""
function covariance_diagonal(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},
    x::AbstractArray{Float64, 2})
    # ...
end

"""
   Optional speedup of `covariance` function that is invoked during training of the GP.
   Intermediate matrices that are re-used between this function and `covariance_grad` could
   be cached in `ker.cache`
"""
function covariance_training(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},
    training_x::AbstractArray{Float64, 2})
    # ...
end

"""
    Optional gradient of `covariance` function with respect to hyperparameters, required
    for optimising with `ConjugateGradient` method. If not provided, `NelderMead` optimiser
    will be used.

    Use `scaled_squared_distance_grad` helper function here.
"""
function covariance_grad(ker::MyCustomkernel, log_theta::AbstractArray{Float64, 1},
    x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    # ...
end

gpm = GPModel(training_x, training_y, MyCustomkernel())
theta = gp_train(gpm)
(test_y, test_var) = gp_regression(test_x, gpm)
```
