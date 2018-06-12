# Examples

* [Basic Gaussian Process Regression](@ref example-1)
* [Optimising Hyperparameters for GP Regression](@ref example-2)

### [Basic Gaussian Process Regression](@id example-1)

```julia
using GpAbc, Distributions, PyPlot

# prepare the data
n = 30
f(x) = x.ˆ2 + 10 * sin.(x) # the latent function

training_x = sort(rand(Uniform(-10, 10), n))
training_y = f(training_x)
training_y += 20 * (rand(n) - 0.5) # add some noise

test_x = collect(linspace(min(training_x...), max(training_x...), 1000))
gpm = GPModel(training_x, training_y) # SE ISO kernel is used by default
set_hyperparameters(gpm, [37.08, 1.0, 6.58]) # see SquaredExponentialIsoKernel reference documentation for description of hyperparameters
(test_y, test_var) = gp_regression(test_x, gpm) # see gp_regression reference documentation  for description of optional parameters
plot(test_x, [test_y f(test)]) # ... and more sophisticated plotting
```

### [Optimising Hyperparameters for GP Regression](@id example-2)
Based on [Basic Gaussian Process Regression](@ref example-1), but with added optimisation
of hyperparameters:
```julia
using GaussProABC
# prepare the data ...
gpm = GPModel(training_x, training_y)
theta_mle = gp_train(gpm) # see gp_train reference documentation for description of optional parameters
(test_y, test_var) = gp_regression(test_x, gpm) # optimised hyperparameters are stored in gpm, so no need to pass them again
```
