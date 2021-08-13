module LshAsymmetricHasher

include("./asym_hash/transformations.jl")
include("./simhash_wrapper.jl")

export AsymHasherParams

using Base.Iterators: partition
using Random: AbstractRNG

using Slide: Float
using Slide.LSH: AbstractHasher, Lsh
using Slide.Hash: LshParams, AbstractLshParams

import Slide.Hash
import Slide.LSH


# Wrapper applying transformations and then calling hasher
struct AsymHasher{H<:AbstractHasher{Vector{Float}},T<:AbstractTransformation} <:
       AbstractHasher{SubArray{Float}}
    hasher::H
    transformation::T
end


function LSH.compute_signatures!(
    signatures::T,
    h::AsymHasher,
    raw_elem::SubArray{Float},
) where {T<:AbstractArray{Int}}
    elem = transform_data(h.transformation, raw_elem)

    h.hasher.compute_signatures!(signatures, h.hasher, elem)
end

function LSH.compute_signatures(h::AsymHasher, raw_elem::SubArray{Float})::Vector{Int}
    signatures = Vector{Int}(undef, h.n_tables)

    LSH.compute_signatures!(signatures, h, raw_elem)

    signatures
end

@inline function LSH.compute_query_signatures(
    h::AsymHasher,
    raw_elem::SubArray{Float},
)::Vector{Int}
    signatures = Vector{Int}(undef, h.n_tables)

    LSH.compute_query_signatures!(signatures, h, raw_elem)

    signatures
end

@inline function LSH.compute_query_signatures!(
    signatures::T,
    h::AsymHasher,
    raw_elem::SubArray{Float},
) where {T<:AbstractArray{Int}}
    elem = transform_query(h.transformation, raw_elem)

    h.hasher.compute_signatures!(signatures, h.hasher, elem)
end

const LshAsymHasher{Id} = Lsh{SubArray{Float},Id,AsymHasher}

struct LshAsymHasherParams{T<:AbstractLshParams} <: AbstractLshParams
    hasher_params::T
    m::Int
end

function Hash.init_lsh!(
    asym_hasher_params::LshAsymHasherParams,
    rng::Rand,
    ::Type{Id},
)::LshAsymHasher where {Id,Rand<:AbstractRNG}
    hasher_params = asym_hasher_params.hasher_params
    lsh_params = hasher_params.lsh_params

    transformation = if typeof(hasher_params) == LshSimHashParams
        MIPStoCosineTransformation(m)
        error("Unrecognized asym hasher params: $(typeof(hasher_params))")
    end

    hasher = init_lsh!(hasher_params, rng, typeof(Id))

    Lsh(
        lsh_params.n_tables,
        lsh_params.n_buckets,
        lsh_params.max_bucket_len,
        AsymHasher(hasher, transformation),
        SubArray{Float},
        Id,
    )
end

function get_asym_hasher_params()::Vector{AsymHasherParams} end

end # LshAsymmetricHasher
