module LshAsymmetricHasher

include("./asym_hash/transformations.jl")

export LshAsymHasherParams

using Base.Iterators: partition
using Random: AbstractRNG

using Slide: Float, FloatVector, LshBatch
using Slide.LSH: AbstractHasher, Lsh
using Slide.Hash: LshParams, AbstractLshParams, init_lsh!
using Slide.LshSimHashWrapper: LshSimHashParams

import Slide.Hash
import Slide.LSH


# Wrapper applying transformations and then calling hasher
mutable struct AsymHasher <: AbstractHasher{FloatVector}
    hasher::AbstractHasher{FloatVector}
    transformation::AbstractTransformation
    n_tables::Int
end

function LSH.compute_signatures!(
    signatures::T,
    h::AsymHasher,
    raw_elem::K,
) where {T<:AbstractArray{Int},K<:FloatVector}
    elem = transform_data(h.transformation, raw_elem)

    LSH.compute_signatures!(signatures, h.hasher, elem)
end

function LSH.compute_signatures(
    h::AsymHasher,
    raw_elem::K,
)::Vector{Int} where {K<:FloatVector}
    signatures = Vector{Int}(undef, h.n_tables)

    LSH.compute_signatures!(signatures, h, raw_elem)

    signatures
end

@inline function LSH.compute_query_signatures(
    h::AsymHasher,
    raw_elem::K,
)::Vector{Int} where {K<:FloatVector}
    signatures = Vector{Int}(undef, h.n_tables)

    LSH.compute_query_signatures!(signatures, h, raw_elem)

    signatures
end

@inline function LSH.compute_query_signatures!(
    signatures::T,
    h::AsymHasher,
    raw_elem::K,
) where {T<:AbstractArray{Int},K<:FloatVector}
    elem = transform_query(h.transformation, raw_elem)

    LSH.compute_signatures!(signatures, h.hasher, elem)
end

const LshAsymHasher{Id} = Lsh{FloatVector,Id,AsymHasher}

mutable struct LshAsymHasherParams <: AbstractLshParams
    hasher_params::AbstractLshParams
    m::Int
    max_norm::Int
end

function Hash.init_lsh!(
    asym_hasher_params::LshAsymHasherParams,
    rng::Rand,
    ::Type{Id},
)::LshAsymHasher where {Id,Rand<:AbstractRNG}
    hasher_params = asym_hasher_params.hasher_params
    lsh_params = hasher_params.lsh_params

    transformation = get_transformation(
        typeof(hasher_params),
        asym_hasher_params.m,
        asym_hasher_params.max_norm,
    )

    hasher = LSH.init_hasher(hasher_params, rng)
    Lsh(
        lsh_params.n_tables,
        lsh_params.n_buckets,
        lsh_params.max_bucket_len,
        AsymHasher(hasher, transformation, lsh_params.n_tables),
        FloatVector,
        Id,
    )
end

end # LshAsymmetricHasher
