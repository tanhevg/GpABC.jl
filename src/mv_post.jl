using Distributions
import Distributions: length, _rand!

type PosteSampler{T<:Real} <: Sampleable{Multivariate,Continuous}
    post::AbstractArray{T,2}
    function PosteSampler{T}(post::AbstractArray{T,2}) where {T}
        return new{T}(post)
    end
end

PosteSampler(post::AbstractArray{T,2}) where {T<:Real} = PosteSampler{T}(post)

length(s::PosteSampler) = size(s.post, 1)

function _rand!(s::PosteSampler{T}, x:::AbstractArray{T,2}) where {T}
    return zeros(size(post,1))
end


M = randn(1000,4)
prior_design=PosteSampler(M)
rand(PosteSampler,M)
