import Distributions: length, _rand!, _logpdf

struct MvUniform{T<:Real} <: ContinuousMultivariateDistribution
    contents::AbstractArray{Uniform{T},1}
end

MvUniform(as::AbstractArray{T, 1}, bs::AbstractArray{T, 1}) where {T<:Real} = MvUniform{T}(Uniform{T}.(as, bs))

function length(d::MvUniform)
    size(d.contents, 1)
end

function _rand!(d::MvUniform{T}, x::AbstractVector{T}) where {T}
    copy!(x,rand.(d.contents))
end

function _logpdf(d::MvUniform{T}, x::AbstractVector{T}) where {T}
    pp = hcat(collect.(params.(d.contents))...)'
    if all(pp[:,1] .<= x .<= pp[:,2])
        return -sum(log.(pp[:, 2] - pp[:, 1]))
    else
        return -T(Inf)
    end
end
