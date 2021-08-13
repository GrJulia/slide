using LinearAlgebra

using Slide: Float
using Slide.LshSimHashWrapper: LshSimHashParams


abstract type AbstractTransformation end

function transform_data(t::AbstractTransformation, data::SubArray{Float})::SubArray{Float}
    error("unimplemented")
end

function transform_query(t::AbstractTransformation, data::SubArray{Float})::SubArray{Float}
    error("unimplemented")
end

struct MIPStoL2NNSTransformation <: AbstractTransformation
    m::Int
end

struct MIPStoCosineTransformation <: AbstractTransformation
    m::Int
end

function get_transformation(::Type{LshSimHashParams}, m::Int)
    return MIPStoCosineTransformation(m)
end

# function get_transformation(::Type{LshL2LshHashParams}, m::Int)
#     return MIPStoCosineTransformation(m)
# end

function transform_data(t::MIPStoL2NNSTransformation, data::SubArray{Float})::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    curr_norm_pow = norm(data)
    for i = data_len+1:data_len+t.m
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    out[data_len+t.m+1:end] .= 0.5

    @view out[:]
end

function transform_query(t::MIPStoL2NNSTransformation, data::SubArray{Float})::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    out[data_len+1:data_len+t.m] .= 0.5

    curr_norm_pow = norm(data)
    for i = data_len+t.m+1:data_len+2*t.m
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    @view out[:]
end

function transform_data(t::MIPStoCosineTransformation, data::SubArray{Float})::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    curr_norm_pow = norm(data)
    for i = data_len+1:data_len+t.m
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    out[data_len+t.m+1:end] .= 0

    @view out[:]
end

function transform_query(
    t::MIPStoCosineTransformation,
    data::SubArray{Float},
)::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    out[data_len+1:data_len+t.m] .= 0

    curr_norm_pow = norm(data)
    for i = data_len+t.m+1:data_len+2*t.m
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    @view out[:]
end
