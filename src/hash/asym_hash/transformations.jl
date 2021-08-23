using LinearAlgebra: norm

using Slide: Float
using Slide.LshSimHashWrapper: LshSimHashParams


abstract type AbstractTransformation end

function transform_data(
    t::T,
    data::K,
)::L where {T<:AbstractTransformation,K<:AbstractArray{Float},L<:AbstractArray{Float}}
    error("unimplemented")
end

function transform_query(
    t::T,
    data::K,
)::L where {T<:AbstractTransformation,K<:AbstractArray{Float},L<:AbstractArray{Float}}
    error("unimplemented")
end

struct MipsToNnsTransformation <: AbstractTransformation
    m::Int
end

struct MipsToCosineTransformation <: AbstractTransformation
    m::Int
end

get_transformation(::Type{LshSimHashParams}, m::Int) = MipsToCosineTransformation(m)

# function get_transformation(::Type{LshL2LshHashParams}, m::Int)
#     return MipsToNnsTransformation(m)
# end

@inline function divide_into_parts(data_len, m)
    base = 1:data_len
    first_tail = data_len+1:data_len+m
    second_tail = data_len+m+1:data_len+2*m
    base, first_tail, second_tail
end

"""
Implementations based on the section 3 of the paper "Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)".
Link: https://papers.nips.cc/paper/2014/file/310ce61c90f3a46e340ee8257bc70e93-Paper.pdf
"""
function transform_data(t::MipsToNnsTransformation, data::SubArray{Float})::Vector{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)

    base, first_tail, second_tail = divide_into_parts(data_len, t.m)

    out[base] = data

    curr_norm_pow = norm(data)
    for i in first_tail
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    out[second_tail] .= 0.5

    out
end

function transform_query(t::MipsToNnsTransformation, data::SubArray{Float})::Vector{Float}
    data_len = length(data)
    out = Vector{Float}(undef, data_len + 2 * t.m)

    base, first_tail, second_tail = divide_into_parts(data_len, t.m)

    out[base] = data

    out[first_tail] .= 0.5

    curr_norm_pow = norm(data)
    for i in second_tail
        curr_norm_pow ^= 2
        out[i] = curr_norm_pow
    end

    out
end

"""
Implementations based on the sections 4 and 5 of the paper "Improved Asymmetric Locality Sensitive Hashing (ALSH) for Maximum Inner Product Search (MIPS)".
Link: https://arxiv.org/pdf/1410.5410.pdf
"""
function transform_data(t::MipsToCosineTransformation, data::SubArray{Float})::Vector{Float}
    data_len = length(data)
    out = zeros(Float, data_len + 2 * t.m)

    base, first_tail, _ = divide_into_parts(data_len, t.m)

    out[base] = data

    curr_norm_pow = norm(data)
    for i in first_tail
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    out
end

function transform_query(
    t::MipsToCosineTransformation,
    data::SubArray{Float},
)::Vector{Float}
    data_len = length(data)
    out = zeros(Float, data_len + 2 * t.m)

    base, _, second_tail = divide_into_parts(data_len, t.m)

    out[base] = data

    curr_norm_pow = norm(data)
    for i in second_tail
        curr_norm_pow ^= 2
        out[i] = 0.5 - curr_norm_pow
    end

    out
end
