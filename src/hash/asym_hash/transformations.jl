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

struct MipsToNnsTransformation <: AbstractTransformation
    m::Int
end

struct MipsToCosineTransformation <: AbstractTransformation
    m::Int
end

function get_transformation(::Type{LshSimHashParams}, m::Int)
    return MipsToCosineTransformation(m)
end

# function get_transformation(::Type{LshL2LshHashParams}, m::Int)
#     return MipsToNnsTransformation(m)
# end

function transform_data(t::MipsToNnsTransformation, data::SubArray{Float})::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)

    base, first_tail, second_tail =
        1:data_len, data_len+1:data_len+t.m, data_len+t.m+1:data_len+2*t.m

    out[base] = data

    curr_norm_pow = norm(data)
    for i in first_tail
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    out[second_tail] .= 0.5

    @view out[:]
end

function transform_query(t::MipsToNnsTransformation, data::SubArray{Float})::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)

    base, first_tail, second_tail =
        1:data_len, data_len+1:data_len+t.m, data_len+t.m+1:data_len+2*t.m

    out[base] = data

    out[first_tail] .= 0.5

    curr_norm_pow = norm(data)
    for i in second_tail
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    @view out[:]
end

function transform_data(
    t::MipsToCosineTransformation,
    data::SubArray{Float},
)::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)

    base, first_tail, second_tail =
        1:data_len, data_len+1:data_len+t.m, data_len+t.m+1:data_len+2*t.m

    out[base] = data

    curr_norm_pow = norm(data)
    for i in first_tail
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    out[second_tail] .= 0

    @view out[:]
end

function transform_query(
    t::MipsToCosineTransformation,
    data::SubArray{Float},
)::SubArray{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)

    base, first_tail, second_tail =
        1:data_len, data_len+1:data_len+t.m, data_len+t.m+1:data_len+2*t.m

    out[base] = data

    out[first_tail] .= 0

    curr_norm_pow = norm(data)
    for i in second_tail
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    @view out[:]
end
