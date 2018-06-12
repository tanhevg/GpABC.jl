# Examples

* [Basic Gaussian Process Regression](@ref example-1)

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
set_hyperparameters(gpm, [37.08, 1.0, 6.58]) # see reference documentation of SquaredExponentialIsoKernel for description of hyperparameters
(test_y, test_var) = gp_regression(test_x, gpm) # see reference documentation of gp_regression for description of optional parameters
plot(test_x, [test_y f(test)]) # ... and more sophisticated plotting
```

**TODO**
