"""
    scaled_squared_distance(log_ell::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2})

Compute the scaled squared distance between `x` and `z`:
```math
r_{ij} = \\sum_{k=1}^d\\frac{(x_{ik}-z_{jk})^2}{l_k^2}
```

# Arguments
- `x, z`: Input data, reshaped into 2-d arrays.
  `x` must have dimensions ``n \\times d``; `z` must have dimensions ``m \\times d``.
- `log_ell`: logarithm of length scale(s). Can either be an array of size one (isotropic),
  or an array of size `d` (ARD)

# Return
An ``n \\times m`` matrix of scaled squared distances
"""
function scaled_squared_distance(log_ell::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    ell = exp.(-log_ell .* 2)
    if x1 === x2
        mu = mean(x1, 1)
    else
        n1 = size(x1, 1)
        n2 = size(x2, 1)
        mu = n1 / (n1 + n2) * mean(x1, 1) + n2 / (n1 + n2) * mean(x2, 1)
    end
    x1 = x1 .- mu               # subtract mean for stability, in case x1 or x2 are huge
    ax1 = x1 .* ell'            # a * x1
    sax1 = sum(x1 .* ax1, 2)    # a * x1 ^ 2
    if x1 === x2                # a shortcut if we need K(x, x) for log likelihood
        ax2 = ax1
        sax2 = sax1
    else                        # repeat computations for x2
        x2 = x2 .- mu
        ax2 = x2 .* ell'
        sax2 = sum(x2 .* ax2, 2)
    end
    D2 = sax1 .- 2 * x1 * ax2' .+ sax2'   # a((b-c)^2) = ab^2 - 2abc + ac^2
end

"""
    scaled_squared_distance_grad(log_ell::AbstractArray{Float64, 1},
        x::AbstractArray{Float64, 2}, z::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})

Return the gradient of the [`scaled_squared_distance`](@ref) function with respect to
logarigthms of length scales, based on the provided direction matrix.

# Arguments
- `x, z`: Input data, reshaped into 2-d arrays.
  `x` must have dimensions ``n \\times d``; `z` must have dimensions ``m \\times d``.
- `log_ell`: logarithm of length scale(s). Can either be an array of size one (isotropic),
  or an array of size `d` (ARD)
- `R` the direction matrix, ``n \\times m``. This can be used to compute the gradient
  of a function that depends on [`scaled_squared_distance`](@ref) via the chain rule.

# Return
A vector of size `length(log_ell)`, whose ``k``'th element is equal to
```math
\\text{tr}(R \\frac{\\partial K}{\\partial l_k})
```
"""
function scaled_squared_distance_grad(log_ell::AbstractArray{Float64, 1},
        x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2}, R::AbstractArray{Float64, 2})
    n1 = size(x1, 1)
    n2 = size(x2, 2)
    if x1 === x2                                        # a shortcut if x1 and x2 are the same - handy for log likelihood optimisation
        x1 = x1 .- mean(x1, 1)                          # subtract mean for stability if x1 is huge
        x12_2 = (sum(R, 2)' + sum(R, 1)) * (x1 .^ 2)    # R * x1 ^ 2 + R * x2 ^ 2
        x1x2 = sum((R * x1) .* x1, 1)                   # R * x1 * x2
        rk = x12_2' - 2 * x1x2'                         # R * (x1 - x2) ^ 2 = R * x1 ^ 2 - 2 * R * x1 * x2 + R * x2 ^ 2
    else
        mu = n1 / (n1 + n2) * mean(x1, 1) + n2 / (n1 + n2) * mean(x2, 1)
        x1 = x1 .- mu                                   # subtract mean for stability if x1 or x2 is huge
        x2 = x2 .- mu
        x1_2 = sum(R, 2)' * (x1 .^ 2)                   # R * x1 ^ 2
        x2_2 = sum(R, 1) * (x2 .^ 2)                    # R * x2 ^ 2
        x1x2 = sum((R * x2) .* x1, 1)                   # R * x1 * x2
        rk = x1_2' - 2 * x1x2' + x2_2'                  # R * (x1 - x2) ^ 2 = R * x1 ^ 2 - 2 * R * x1 * x2 + R * x2 ^ 2
    end
    ell = exp.(-2 * log_ell)
    return reshape(-2 * ell .* rk, length(ell))
end
