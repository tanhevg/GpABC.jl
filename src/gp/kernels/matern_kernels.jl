mutable struct MaternCovarianceCache
    last_theta::AbstractArray{Float64, 1}
    D::AbstractArray{Float64, 2}
    D2::AbstractArray{Float64, 2}
    K::AbstractArray{Float64, 2}
end

"""
    MaternIsoKernel <: AbstractGPKernel

Matérn kernel with uniform length scale across all dimensions, ``l``.
Parameter ``\\nu`` (nu) is passed in constructor. Currently, only values of
``\\nu=1``, ``\\nu=3`` and ``\\nu=5`` are supported.

```math
\\begin{aligned}
K_{\\nu=1}(r) &= \\sigma_f^2e^{-\\sqrt{r}}\\\\
K_{\\nu=3}(r) &= \\sigma_f^2(1 + \\sqrt{3r})e^{-\\sqrt{3r}}\\\\
K_{\\nu=5}(r) &= \\sigma_f^2(1 + \\sqrt{3r} + \\frac{5}{3}r)e^{-\\sqrt{5r}}\\\\
r_{ij} &= \\sum_{k=1}^d\\frac{(x_{ik}-z_{jk})^2}{l^2}
\\end{aligned}
```

``r_{ij}`` are computed by [`scaled_squared_distance`](@ref)

# Hyperparameters
Hyperparameters vector for this kernel must contain two elements, in the following order:
1. ``\\sigma_f``: the signal standard deviation
2. ``l``: the length scale
"""
struct MaternIsoKernel <: AbstractGPKernel
    nu::Int
    cache::MaternCovarianceCache
end

function check_nu(nu::Int)
    if nu != 1 && nu != 3 && nu != 5
        error("Unsupported ν = $(nu)")
    end
end

function MaternIsoKernel(nu::Int)
    check_nu(nu)
    MaternIsoKernel(nu, MaternCovarianceCache(zeros(0), zeros(0,0), zeros(0,0), zeros(0,0)))
end

"""
    ExponentialIsoKernel

Alias for [`MaternIsoKernel`](@ref)(1)
"""
ExponentialIsoKernel() = MaternIsoKernel(1)

"""
    MaternArdKernel <: AbstractGPKernel

Matérn kernel with distinct length scale for each dimention, ``l_k``.
Parameter ``\\nu`` (nu) is passed in constructor. Currently, only values of
``\\nu=1``, ``\\nu=3`` and ``\\nu=5`` are supported.

```math
\\begin{aligned}
K_{\\nu=1}(r) &= \\sigma_f^2e^{-\\sqrt{r}}\\\\
K_{\\nu=3}(r) &= \\sigma_f^2(1 + \\sqrt{3r})e^{-\\sqrt{3r}}\\\\
K_{\\nu=5}(r) &= \\sigma_f^2(1 + \\sqrt{3r} + \\frac{5}{3}r)e^{-\\sqrt{5r}}\\\\
r_{ij} &= \\sum_{k=1}^d\\frac{(x_{ik}-z_{jk})^2}{l_k^2}
\\end{aligned}
```

``r_{ij}`` are computed by [`scaled_squared_distance`](@ref)

# Hyperparameters
The length of hyperparameters array for this kernel depends on the dimensionality
of the data. Assuming each data point is a vector in a ``d``-dimensional space,
this kernel needs ``d+1`` hyperparameters, in the following order:
1. ``\\sigma_f``: the signal standard deviation
2. ``l_1, \\ldots, l_d``: the length scales for each dimension
"""
struct MaternArdKernel <: AbstractGPKernel
    nu::Int
    cache::MaternCovarianceCache
end

"""
    ExponentialArdKernel

Alias for [`MaternArdKernel`](@ref)(1)
"""
ExponentialArdKernel() = MaternArdKernel(1)

function MaternArdKernel(nu::Int)
    check_nu(nu)
    MaternArdKernel(nu, MaternCovarianceCache(zeros(0), zeros(0,0), zeros(0,0), zeros(0,0)))
end


function get_hyperparameters_size(kernel::MaternIsoKernel, training_data::AbstractArray{Float64, 2})
    return 2
end

function get_hyperparameters_size(kernel::MaternArdKernel, training_data::AbstractArray{Float64, 2})
    return size(training_data, 2) + 1
end


function matern_poly(nu, D, D2)
    if nu == 1
        return 1
    elseif nu == 3
        return D .* sqrt(3) .+ 1
    elseif nu == 5
        return D2 .* (5/3) .+ D * sqrt(5) .+ 1
    end
end

function matern_iso_poly_grad(nu, D, D2)
    if nu == 1
        return D
    elseif nu == 3
        return 3 * D2
    elseif nu == 5
        return D2 .* (5/3) .* (D .* sqrt(5) .+ 1)
    end
end

function matern_ard_poly_grad(nu, D)
    if nu == 1
        ret = -1./D
        ret[find(D .== 0.0)] = 0.0
        ret
    elseif nu == 3
        -sqrt(3)
    elseif nu == 5
        -(sqrt(5) .* D .+ 1) .* (sqrt(5) / 3)
    end
end

function matern_covariance_common(nu::Int, cache::MaternCovarianceCache, log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    sigma_f = exp(log_theta[1] * 2)
    D2 = scaled_squared_distance(log_theta[2:end], x1, x2)
    D2[find(D2 .< 0)] = 0.0  # could be negative due to numerical noise
    D = sqrt.(D2)
    K = sigma_f .* matern_poly(nu, D, D2) .* exp.(-sqrt(nu) .* D)
end

function covariance(ker::MaternIsoKernel, log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    matern_covariance_common(ker.nu, ker.cache, log_theta, x1, x2)
end

function covariance(ker::MaternArdKernel, log_theta::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    matern_covariance_common(ker.nu, ker.cache, log_theta, x1, x2)
end

function matern_covariance_common_training(nu::Int, cache::MaternCovarianceCache,
    log_theta::AbstractArray{Float64, 1}, x::AbstractArray{Float64, 2})
    if log_theta != cache.last_theta ||
            size(cache.D, 1) != size(x, 1) ||
            size(cache.D2, 1) != size(x, 1) ||
            size(cache.K, 1) != size(x, 1)
        sigma_f = exp(log_theta[1] * 2)
        D2 = scaled_squared_distance(log_theta[2:end], x, x)
        D2[find(D2 .< 0)] = 0.0  # could be negative due to numerical noise
        D = sqrt.(D2)
        K = sigma_f .* matern_poly(nu, D, D2) .* exp.(-sqrt(nu) .* D)
        cache.last_theta = copy(log_theta)
        cache.D = D
        cache.D2 = D2
        cache.K = K
    end
    cache.K
end

function covariance_training(ker::MaternIsoKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    matern_covariance_common_training(ker.nu, ker.cache, log_theta, x)
end

function covariance_training(ker::MaternArdKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    matern_covariance_common_training(ker.nu, ker.cache, log_theta, x)
end

function covariance_diagonal(ker::MaternIsoKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    fill(exp(log_theta[1] * 2), (size(x, 1), 1))
end

function covariance_diagonal(ker::MaternArdKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2})
    fill(exp(log_theta[1] * 2), (size(x, 1), 1))
end


function matern_covariance_grad_common(nu::Int, cache::MaternCovarianceCache, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    sigma_f = exp(log_theta[1] * 2)
    if log_theta == cache.last_theta &&
            size(cache.D, 1) == size(x, 1) &&
            size(cache.D2, 1) != size(x, 1) &&
            size(cache.K, 1) != size(x, 1)
        K = cache.K
        D = cache.D
        D2 = cache.D2
    else
        D2 = scaled_squared_distance(log_theta[2:end], x, x)
        D2[find(D2 .< 0)] = 0.0  # could be negative due to numerical noise
        D = sqrt.(D2)
        K = sigma_f .* matern_poly(nu, D, D2) .* exp.(-sqrt(nu) .* D)
        cache.last_theta = copy(log_theta)
        cache.D = D
        cache.D2 = D2
        cache.K = K
    end
    KR = K .* R
    d_sigma_f = 2 * sum(KR)
    exp_nu_D_R = exp.(-sqrt(nu) .* D) .* R
    if length(log_theta) == 2
        d_ell = sum(matern_iso_poly_grad(nu, D, D2) .* exp_nu_D_R)
    else
        d_ell = scaled_squared_distance_grad(log_theta[2:end], x, x,
            sqrt(nu) / 2 .* exp_nu_D_R .* matern_ard_poly_grad(nu, D))
    end
    return [d_sigma_f; sigma_f * d_ell]
end

function covariance_grad(ker::MaternIsoKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    matern_covariance_grad_common(ker.nu, ker.cache, log_theta, x, R)
end

function covariance_grad(ker::MaternArdKernel, log_theta::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    matern_covariance_grad_common(ker.nu, ker.cache, log_theta, x, R)
end
