mutable struct HPOptimisationCache
    theta::AbstractArray{Float64, 1}    # hyperparameters
    K::AbstractArray{Float64, 2}        # covariance matrix
    B::AbstractArray{Float64, 2}        # I + K / σ_n ^ 2
    Q::AbstractArray{Float64, 2}        # inv(B)
    L::AbstractArray{Float64, 2}        # chol(B)
    alpha::AbstractArray{Float64, 2}    # inv(B) * y
    R::AbstractArray{Float64, 2}        # α * α' - Q / σ_n ^ 2
end

HPOptimisationCache() = HPOptimisationCache(
        zeros(0),       # theta
        zeros(0, 0),    # K
        zeros(0, 0),    # B
        zeros(0, 0),    # Q
        zeros(0, 0),    # L
        zeros(0, 0),    # alpha
        zeros(0, 0))    # R

abstract type AbstractGaussianProcess end;

"""
    GPModel

The main type that is used by most functions within the package.

All data matrices are row-major.

# Fields
- `kernel::AbstractGPKernel`: the kernel
- `gp_training_x::AbstractArray{Float64, 2}`: training `x`. Size: ``n \\times d``.
- `gp_training_y::AbstractArray{Float64, 2}`: training `y`. Size: ``n \\times 1``.
- `gp_test_x::AbstractArray{Float64, 2}`: test `x`.  Size: ``m \\times d``.
- `gp_hyperparameters::AbstractArray{Float64, 1}`: kernel hyperparameters, followed by
  standard deviation of intrinsic noise ``\\sigma_n``, which is always the last element in the array.
- `cache::HPOptimisationCache`: cache of matrices that can be re-used between calls to
  `gp_loglikelihood` and `gp_loglikelihood_grad`
"""
mutable struct GPModel <: AbstractGaussianProcess
    kernel::AbstractGPKernel

    cache::HPOptimisationCache

    gp_training_x::AbstractArray{Float64, 2}
    gp_training_y::AbstractArray{Float64, 2}
    gp_test_x::AbstractArray{Float64, 2}

    gp_hyperparameters::AbstractArray{Float64, 1}
end

"""
    GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
            training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}
            [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])

Default constructor of [`GPModel`](@ref), that will use [`SquaredExponentialIsoKernel`](@ref).
Arguments that are passed as 1-d vectors will be reshaped into 2-d.
"""
function GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
        test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0))
    GPModel(training_x=training_x, training_y=training_y, test_x=test_x)
end

"""
    GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
            training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
            kernel::AbstractGPKernel
            [,test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0)])

Constructor of [`GPModel`](@ref) that allows the kernel to be specified.
Arguments that are passed as 1-d vectors will be reshaped into 2-d.
"""
function GPModel(training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}},
        kernel::AbstractGPKernel,
        test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0))
    GPModel(training_x=training_x, training_y=training_y, kernel=kernel, test_x=test_x)
end

"""
    GPModel(;training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),
        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),
        test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),
        kernel::AbstractGPKernel=SquaredExponentialIsoKernel(),
        gp_hyperparameters::AbstractArray{Float64, 1}=Array{Float64}(0))

Constructor of [`GPModel`](@ref) with explicit arguments.
Arguments that are passed as 1-d vectors will be reshaped into 2-d.
"""
function GPModel(;training_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),
        training_y::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),
        test_x::Union{AbstractArray{Float64, 2}, AbstractArray{Float64, 1}}=zeros(0,0),
        kernel::AbstractGPKernel=SquaredExponentialIsoKernel(),
        gp_hyperparameters::AbstractArray{Float64, 1}=Array{Float64}(0))

    if ndims(training_x) == 1
        training_x = reshape(training_x, length(training_x), 1)
    end
    if size(training_x, 1) < size(training_x, 2)
        warn("GPModel: got a $(size(training_x)) size training array. ",
            "Are you using column major data instead of row major?")
    end
    if ndims(training_y) == 1
        training_y = reshape(training_y, length(training_y), 1)
    end
    if ndims(test_x) == 1
        test_x = reshape(test_x, length(test_x), 1)
    end
    if size(test_x, 1) > 0 && size(training_x, 1) > 0 && size(test_x, 2) != size(training_x, 2)
        error("Test and training x should have the same second dimension")
    end
    if size(training_y, 1) > 0 && size(training_y, 2) != 1
        error("Training y should have second dimension == 1")
    end
    expected_hypers_size = get_hyperparameters_size(kernel, training_x) + 1
    if length(gp_hyperparameters) == 0
        gp_hyperparameters = ones(expected_hypers_size)
    end
    if length(gp_hyperparameters) != expected_hypers_size
        error("Incorrect size of initial hyperparameters vector for ",
            "$(typeof(kernel)): $(length(hyperparameters)). ",
            "Expected $(expected_hypers_size).")
    end
    GPModel(kernel,
        HPOptimisationCache(),
        training_x, training_y, test_x,
        gp_hyperparameters)
end;

function get_hyperparameters_size(gpem::GPModel)
    return get_hyperparameters_size(gpem.kernel, gpem.gp_training_x) + 1
end

"""
    set_hyperparameters(gpm::GPModel, hypers::AbstractArray{Float64, 1})

Set the hyperparameters of the [`GPModel`](@ref)
"""
function set_hyperparameters(gpem::GPModel, hypers::AbstractArray{Float64, 1})
    expected_hypers_size = get_hyperparameters_size(gpem)
    if length(hypers) != expected_hypers_size
        error("Incorrect size of hyperparameters vector for ",
            "$(typeof(gpem.kernel)): $(length(gpem.gp_hyperparameters)). ",
            "Expected $(expected_hypers_size).")
    end
    gpem.gp_hyperparameters = hypers
end

"""
    gp_loglikelihood_log(theta::AbstractArray{Float64, 1}, gpm::GPModel)

Log likelihood function with log hyperparameters. This is the target function of the
hyperparameters optimisation procedure. Its gradient is coputed by [`gp_loglikelihood_grad`](@ref).
"""
function gp_loglikelihood_log(theta::AbstractArray{Float64, 1}, gpem::GPModel)
    cache = gpem.cache
    update_cache!(gpem.kernel, cache, theta, gpem.gp_training_x, gpem.gp_training_y)
    sigma_n2_inv = exp(-2*theta[end])
    x = gpem.gp_training_x
    y = gpem.gp_training_y

    n = size(x, 1)
    ret = -0.5 * dot(y, cache.alpha) - sum(log.(diag(cache.L))) - 0.5 * n * log(2*π/sigma_n2_inv)
end

"""
    gp_loglikelihood(gpm::GPModel)

Compute the log likelihood function, based on the kernel and training data specified in `gpm`.
```math
log p(y \\vert X, \\theta) = - \\frac{1}{2}(y^TK^{-1}y + log \\vert K \\vert + n log 2 \\pi)
```
"""
gp_loglikelihood(gpem::GPModel) = gp_loglikelihood_log(log.(gpem.gp_hyperparameters), gpem)

"""
    gp_loglikelihood(theta::AbstractArray{Float64, 1}, gpm::GPModel)
"""
gp_loglikelihood(theta::AbstractArray{Float64, 1}, gpem::GPModel) = gp_loglikelihood_log(log.(theta), gpem)

"""
    gp_loglikelihood_grad(theta::AbstractArray{Float64, 1}, gpem::GPModel)

Gradient of the log likelihood function ([`gp_loglikelihood_log`](@ref))
with respect to logged hyperparameters.
"""
function gp_loglikelihood_grad(theta::AbstractArray{Float64, 1}, gpem::GPModel)
    cache = gpem.cache
    update_cache!(gpem.kernel, cache, theta, gpem.gp_training_x, gpem.gp_training_y)
    sigma_n2_inv = exp(-2*theta[end])
    x = gpem.gp_training_x
    y = gpem.gp_training_y
    n = size(x, 1)
    kernel_grad = covariance_grad(gpem.kernel, theta[1:end-1], x, cache.R) ./ 2
    if kernel_grad === :covariance_gradient_not_implemented
        return kernel_grad
    end
    dW = cache.Q .* cache.K
    d_sigma_n = sum(dW) * sigma_n2_inv - n + dot(cache.alpha, cache.alpha) / sigma_n2_inv
    push!(kernel_grad, d_sigma_n)
end

"""
    gp_regression(gpm::GPModel; <optional keyword arguments>)

Run the Gaussian Process Regression.

# Arguments
- `gpm`: the [`GPModel`](@ref), that contains the training data (x and y),
  the kernel, the hyperparameters and the test data for running the regression.
- `test_x`: if specified, overrides the test x in `gpm`. Size ``m \\times d``.
- `log_level::Int` (optional): log level. Default is `0`, which is no logging at all. `1`
  makes `gp_regression` print basic information to standard output.
- `full_covariance_matrix::Bool` (optional): whether we need the full
  covariance matrix, or just the variance vector. Defaults to `false` (i.e. just the variance).
- `batch_size::Int` (optional): If `full_covariance_matrix` is set to `false`,
  then the mean and variance vectors will be computed in batches of this size,
  to avoid allocating huge matrices. Defaults to 1000.
- `observation_noise::Bool` (optional): whether the observation noise (with variance
  ``\\sigma_n^2``) should be included in the output variance. Defaults to `true`.

# Return
A tuple of `(mean, var)`. `mean` is a mean vector of the output multivariate Normal
distribution, of size ``m``. `var` is either the covariance matrix of size ``m \\times m``,
or a variance vector of size ``m``, depending on `full_covariance_matrix` flag.
"""
function gp_regression(gpem::GPModel; kwargs...)
    gp_regression_log(log.(gpem.gp_hyperparameters), gpem.gp_test_x, gpem; kwargs...)
end

"""
    gp_regression(test_x::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}},
        gpem::GPModel; <optional keyword arguments>)
"""
function gp_regression(test_x::Union{AbstractArray{Float64, 1}, AbstractArray{Float64, 2}}, gpem::GPModel; kwargs...)
    if ndims(test_x) == 1
        test_x = reshape(test_x, length(test_x), 1)
    end
    gp_regression_log(log.(gpem.gp_hyperparameters), test_x, gpem; kwargs...)
end

function gp_regression_theta(theta::AbstractArray{Float64, 1}, gpem::GPModel; kwargs...)
    gpem.gp_hyperparameters = theta
    gp_regression_log(log.(theta), gpem.gp_test_x, gpem; kwargs...)
end


function gp_regression_log(theta::AbstractArray{Float64, 1},
        x_s::AbstractArray{Float64, 2}, gpem::GPModel;
        log_level::Int=0,
        full_covariance_matrix::Bool=false,
        batch_size::Int=1000,
        observation_noise::Bool=true)
    if (log_level > 0)
        println("Regression log hyperparameters = $(theta)")
    end
    gpem.gp_test_x = x_s

    theta_kernel = theta[1:end-1]
    sigma_n_inv = exp(-theta[end])
    n_s = size(x_s, 1)

    expected_hypers_size = get_hyperparameters_size(gpem)
    if length(theta) != expected_hypers_size
        error("Incorrect size of hyperparameters vector for ",
            "$(typeof(gpem.kernel)): $(length(theta)). ",
            "Expected $(expected_hypers_size).")
    end
    cache = gpem.cache
    update_cache!(gpem.kernel, cache, theta, gpem.gp_training_x, gpem.gp_training_y)

    xs_idx_start = 1
    multiple_batches = !full_covariance_matrix && n_s > batch_size
    xs_idx_end = full_covariance_matrix ? n_s : min(n_s, batch_size)
    ret_mean = Array{Float64}(n_s)
    ret_var = Array{Float64}(n_s)
    batch_counter = 1
    while xs_idx_start <= n_s
        if log_level > 0
            println("Regression batch #$(batch_counter) $(xs_idx_start):$(xs_idx_end)")
            batch_counter += 1
        end
        batch = x_s[xs_idx_start:xs_idx_end, :]
        K_s = covariance(gpem.kernel, theta_kernel, gpem.gp_training_x, batch)'
        v = cache.L' \ (sigma_n_inv * K_s')
        batch_mean = K_s * cache.alpha
        batch_mean = reshape(batch_mean, length(batch_mean))
        if multiple_batches || !full_covariance_matrix
            K_ss = covariance_diagonal(gpem.kernel, theta_kernel, batch)
            if observation_noise
                K_ss += ones(size(batch, 1), 1) * exp(2 * theta[end])
            end
            batch_var = K_ss - sum(v .^ 2, 1)'
            batch_var = reshape(batch_var, length(batch_var))
        else
            K_ss = covariance(gpem.kernel, theta_kernel, batch, batch)
            if observation_noise
                K_ss += I * exp(2 * theta[end])
            end
            batch_var = K_ss - v' * v
        end

        if multiple_batches
            ret_mean[xs_idx_start:xs_idx_end] = batch_mean
            ret_var[xs_idx_start:xs_idx_end] = batch_var
            xs_idx_start = xs_idx_end + 1
            xs_idx_end = min(n_s, xs_idx_start + batch_size - 1)
        else
            return batch_mean, batch_var
        end
    end
    return (ret_mean, ret_var)
end

function update_cache!(kernel::AbstractGPKernel, cache::HPOptimisationCache,
        theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 2})
    if cache.theta != theta
        n = size(x, 1)
        if size(cache.theta) != size(theta)
            cache.theta = copy(theta)
        else
            copy!(cache.theta, theta)
        end
        sigma_n2_inv = exp(-2*theta[end])
        cache.K = covariance_training(kernel, theta[1:end-1], x)
        cache.B = cache.K * sigma_n2_inv + I
        cache.L = chol(Hermitian(cache.B))
        cache.Q = cache.L \ (cache.L' \ I)
        cache.alpha = (cache.L \ (cache.L' \ y)) .* sigma_n2_inv
        cache.R = cache.alpha * cache.alpha' - cache.Q .* sigma_n2_inv
    end
    0
end
