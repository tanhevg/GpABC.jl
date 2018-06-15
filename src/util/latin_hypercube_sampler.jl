import Distributions: length, _rand!

struct LatinHypercubeSampler{T<:Real} <: Sampleable{Multivariate,Continuous}
    mins::AbstractArray{T, 1}
    maxs::AbstractArray{T, 1}
    function LatinHypercubeSampler{T}(mins::AbstractArray{T, 1}, maxs::AbstractArray{T, 1}) where {T}
        size(mins, 1) == size(maxs, 1) ? new{T}(mins, maxs) : error("size mismatch")
    end
end

LatinHypercubeSampler(mins::AbstractArray{T, 1}, maxs::AbstractArray{T, 1}) where {T<:Real} = LatinHypercubeSampler{T}(mins, maxs)

length(s::LatinHypercubeSampler) = size(s.mins, 1)

function _rand!(s::LatinHypercubeSampler{T}, x::AbstractVector{T}) where {T}
    x[:] = s.mins .+ (s.maxs - s.mins) .* rand(size(s.mins, 1))
end

function _rand!(s::LatinHypercubeSampler{T}, m::DenseMatrix{T}) where {T}
    num_samples = size(m, 2)
    dims = size(m, 1)
    @inbounds for i in 1:dims
        interval_len = (s.maxs[i] - s.mins[i]) / num_samples
        m[i, :] = shuffle!(linspace(s.mins[i], s.maxs[i] - interval_len, num_samples) +
                               interval_len*rand(num_samples))
    end
    m
end
