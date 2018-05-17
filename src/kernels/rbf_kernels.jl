mutable struct RbfCovarianceCache
    last_theta::AbstractArray{Float64, 1}
    D2::AbstractArray{Float64, 2}
    K::AbstractArray{Float64, 2}
end

"""
    SquaredExponentialIsoKernel <: AbstractGPKernel

Squared exponential kernel with uniform length scale across all dimensions, ``l``.
```math
K(r) = \\sigma_f^2 e^{-r/2}; r_{ij} = \\sum_{k=1}^d\\frac{(x_{ik}-z_{jk})^2}{l^2}
```

# Hyperparameters
Hyperparameters vector for this kernel must contain two elements, in the following order:
1. ``\\sigma_f``: the signal standard deviation
2. ``l``: the length scale
"""
struct SquaredExponentialIsoKernel <: AbstractGPKernel
    cache::RbfCovarianceCache
end

function SquaredExponentialIsoKernel()
    SquaredExponentialIsoKernel(RbfCovarianceCache(Array{Float64}(0), Array{Float64}(0, 0), Array{Float64}(0, 0)))
end

"""
    SquaredExponentialArdKernel <: AbstractGPKernel

Squared exponential kernel with distinct length scale for each dimention, ``l_k``.
```math
K(r) = \\sigma_f^2 e^{-r/2}; r_{ij} = \\sum_{k=1}^d\\frac{(x_{ik}-z_{jk})^2}{l_k^2}
```

# Hyperparameters
The length of hyperparameters array for this kernel depends on the dimensionality
of the data. Assuming each data point is a vector in a ``d``-dimensional space,
this kernel needs ``d+1`` hyperparameters, in the following order:
1. ``\\sigma_f``: the signal standard deviation
2. ``l_1, \\ldots, l_d``: the length scales for each dimension
"""
struct SquaredExponentialArdKernel <: AbstractGPKernel
    cache::RbfCovarianceCache
end

function SquaredExponentialArdKernel()
    SquaredExponentialArdKernel(RbfCovarianceCache(Array{Float64}(0), Array{Float64}(0, 0), Array{Float64}(0, 0)))
end

function get_hyperparameters_size(kernel::SquaredExponentialIsoKernel, training_data::AbstractArray{Float64, 2})
    return 2
end

function get_hyperparameters_size(kernel::SquaredExponentialArdKernel, training_data::AbstractArray{Float64, 2})
    return size(training_data, 2) + 1
end

function rbf_covariance_grad_common(cache::RbfCovarianceCache, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    if log_theta == cache.last_theta
        K = cache.K
        D2 = cache.D2
    else
        sigma_f = exp(log_theta[1] * 2)
        D2 = scaled_squared_distance(log_theta[2:end], x, x)
        K = sigma_f .* exp.(-D2 ./ 2)
        cache.last_theta = copy(log_theta)
        cache.D2 = D2
        cache.K = K
    end
    KR = K .* R
    d_sigma_f = 2 * sum(KR)
    if length(log_theta) == 2
        d_ell = KR[:]' * D2[:]
    else
        d_ell = -scaled_squared_distance_grad(log_theta[2:end], x, x, KR)/2
    end
    return [d_sigma_f; d_ell]
end

function rbf_covariance_common(log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    sigma_f = exp(log_theta[1] * 2)
    D2 = scaled_squared_distance(log_theta[2:end], x1, x2)
    K = sigma_f .* exp.(-D2 ./ 2)
end

function covariance(ker::SquaredExponentialIsoKernel, log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    rbf_covariance_common(log_theta, x1, x2)
end

function covariance(ker::SquaredExponentialArdKernel, log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    rbf_covariance_common(log_theta, x1, x2)
end

function rbf_covariance_common_training(cache::RbfCovarianceCache,
        log_theta::AbstractArray{Float64, 1}, x::AbstractArray{Float64, 2})
    if log_theta != cache.last_theta
        cache.last_theta = copy(log_theta)
        sigma_f = exp(log_theta[1] * 2)
        D2 = scaled_squared_distance(log_theta[2:end], x, x)
        cache.D2 = D2
        cache.K = sigma_f .* exp.(-D2 ./ 2)
    end
    cache.K
end

function covariance_training(ker::SquaredExponentialIsoKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    rbf_covariance_common_training(ker.cache, log_theta, x)
end

function covariance_training(ker::SquaredExponentialArdKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    rbf_covariance_common_training(ker.cache, log_theta, x)
end

function covariance_diagonal(ker::SquaredExponentialIsoKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    fill(exp(log_theta[1] * 2), (size(x, 1), 1))
end

function covariance_diagonal(ker::SquaredExponentialArdKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    fill(exp(log_theta[1] * 2), (size(x, 1), 1))
end


function covariance_grad(ker::SquaredExponentialIsoKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    rbf_covariance_grad_common(ker.cache, log_theta, x, R)
end

function covariance_grad(ker::SquaredExponentialArdKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    rbf_covariance_grad_common(ker.cache, log_theta, x, R)
end
