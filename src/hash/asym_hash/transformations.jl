using LinearAlgebra

using Slide: Float


abstract type AbstractTransformation end

function transform_data(t::AbstractTransformation, data::SubArray{Float})::Vector{Float}
    error("unimplemented")
end

function transform_query(t::AbstractTransformation, data::SubArray{Float})::Vector{Float}
    error("unimplemented")
end

struct MIPStoL2NNSTransformation <: AbstractTransformation
    m::Int
end

struct MIPStoCosineTransformation <: AbstractTransformation
    m::Int
end

function transform_data(t::MIPStoL2NNSTransformation, data::SubArray{Float})::Vector{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    curr_norm_pow = norm(data)
    for i = data_len+1:data_len+t.m
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    out[data_len+t.m+1:end] .= 0.5

    out
end

function transform_query(t::MIPStoL2NNSTransformation, data::SubArray{Float})::Vector{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    out[data_len+1:data_len+t.m] .= 0.5

    curr_norm_pow = norm(data)
    for i = data_len+t.m+1:data_len+2*t.m
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    out
end

function transform_data(t::MIPStoCosineTransformation, data::SubArray{Float})::Vector{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    curr_norm_pow = norm(data)
    for i = data_len+1:data_len+t.m
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    out[data_len+t.m+1:end] .= 0

    out
end

function transform_query(
    t::MIPStoCosineTransformation,
    data::SubArray{Float},
)::Vector{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)
    out[1:data_len] = data

    out[data_len+1:data_len+t.m] .= 0

    curr_norm_pow = norm(data)
    for i = data_len+t.m+1:data_len+2*t.m
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    out
end
