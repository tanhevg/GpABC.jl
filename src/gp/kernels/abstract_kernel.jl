"""
    AbstractGPKernel

Abstract kernel type. User-defined kernels should derive from it.

Implementations have to provide methods for [`get_hyperparameters_size`](@ref)
and [`covariance`](@ref). Methods for [`covariance_training`](@ref), [`covariance_diagonal`](@ref)
and [`covariance_grad`](@ref) are optional.
"""
abstract type AbstractGPKernel end

"""
    covariance(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})

Return the covariance matrix. Should be overridden by kernel implementations.

# Arguments
- `ker`: The kernel object. Implementations must override with their own subtype.
- `log_theta`: natural logarithm of hyperparameters.
- `x, z`: Input data, reshaped into 2-d arrays.
  `x` must have dimensions ``n \\times d``; `z` must have dimensions ``m \\times d``.

# Return
The covariance matrix, of size ``n \\times m``.
"""
function covariance(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2}) end

"""
    covariance_training(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        training_x::AbstractArray{Float64, 2})

This is a speedup version of [`covariance`](@ref), which is only called during
traing sequence. Intermediate matrices computed
in this function for particular hyperparameters can be cached and reused subsequently, either
in this function or in [`covariance_grad`](@ref)

Default method just delegates to [`covariance`](@ref) with `x === z`. Kernel implementations can
optionally override it for betrer performance.

See [`covariance`](@ref) for description of arguments and return values.
"""
function covariance_training(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    covariance(ker, log_theta, x, x)
end

"""
    covariance_diagonal(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})

This is a speedup version of [`covariance`](@ref), which is invoked if the caller
is not interested in the entire covariance matrix, but only needs the variance, i.e. the
diagonal of the covariance matrix.

Default method just returns `diag(covariance(...))`, with `x === z`. Kernel implementations can
optionally override it to achieve betrer performance, by not computing the non diagonal
elements of covariance matrix.

See [`covariance`](@ref) for description of arguments.

# Return
The 1-d array of variances, of size `size(x, 1)`.
"""
function covariance_diagonal(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    diag(covariance(ker, log_theta, x, x))
end

"""
    covariance_grad(ker::AbstractGPKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})

Return the gradient of the covariance function with respect to logarigthms of hyperparameters,
based on the provided direction matrix.

This function can be optionally overridden by kernel implementations.
If the gradient function is not provided, [`gp_train`](@ref) will fail back to
`NelderMead` algorithm by default.

# Arguments
- `ker`: The kernel object. Implementations must override with their own subtype.
- `log_theta`:  natural logarithm of hyperparameters
- `x`: Training data, reshaped into a 2-d array.
  `x` must have dimensions ``n \\times d``.
- `R` the directional matrix, ``n \\times n``
```math
R = \\frac{1}{\\sigma_n^2}(\\alpha * \\alpha^T - K^{-1}); \\alpha = K^{-1}y
```

# Return
A vector of size `length(log_theta)`, whose ``j``'th element is equal to
```math
tr(R \\frac{\\partial K}{\\partial \\eta_j})
```
"""
function covariance_grad(ker::AbstractGPKernel, theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    :covariance_gradient_not_implemented
end

"""
    get_hyperparameters_size(kernel::AbstractGPKernel, training_data::AbstractArray{Float64, 2})

Return the number of hyperparameters used by the kernel on the training data set.
Should be overridden by kernel implementations.
"""
function get_hyperparameters_size(kernel::AbstractGPKernel, training_data::AbstractArray{Float64, 2}) end
