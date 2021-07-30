
module LshSimHashWrapper

export LshSimHashParams

using Base.Iterators: partition
using Random: AbstractRNG

using Slide: Float
using Slide.LSH: AbstractHasher, Lsh, add!, retrieve
using Slide.SimHash: SimHasher, Signature, initialize!, signature, signature_len
using Slide.Hash: LshParams, AbstractLshParams

import Slide.Hash
import Slide.LSH


struct SimhasherWrapper <: AbstractHasher{SubArray{Float}}
    signature_len::UInt8
    hasher::SimHasher

    function SimhasherWrapper(signature_len::UInt8, hasher::SimHasher)
        signature_len >= 8 * sizeof(Int) && error(
            "Signature length can't be greater than $(8 * sizeof(Int)), got $signature_len",
        )
        new(signature_len, hasher)
    end
end


@inline function compute_integer_from_bin(
    bin_repr::A,
)::Int where {T<:Number,A<:AbstractVector{T}}
    foldl(bin_repr, init = zero(T)) do acc, n
        2 * acc + n
    end |> Int
end

function LSH.compute_signatures!(
    signatures::T,
    h::SimhasherWrapper,
    elem::SubArray{Float},
) where {T<:AbstractArray{Int}}

    raw_signature = signature(h.hasher, elem)
    raw_signature_chunks = partition(raw_signature, h.signature_len)

    @inbounds for (i, array) in enumerate(raw_signature_chunks)
        signatures[i] = compute_integer_from_bin(array)
    end
end

function LSH.compute_signatures(h::SimhasherWrapper, elem::SubArray{Float})::Vector{Int}
    n_signatures = signature_len(h.hasher) ÷ h.signature_len
    signatures = Vector{Int}(undef, n_signatures)

    LSH.compute_signatures!(signatures, h, elem)

    signatures
end

@inline function LSH.compute_query_signatures(
    h::SimhasherWrapper,
    elem::SubArray{Float},
)::Vector{Int}
    LSH.compute_signatures(h, elem)
end

@inline function LSH.compute_query_signatures!(
    signatures::T,
    h::SimhasherWrapper,
    elem::SubArray{Float},
) where {T<:AbstractArray{Int}}
    LSH.compute_signatures!(signatures, h, elem)
end


const LshSimHash{Id} = Lsh{SubArray{Float},Id,SimhasherWrapper}

struct LshSimHashParams <: AbstractLshParams
    lsh_params::LshParams
    vector_len::Int
    signature_len::Int
    sample_size::Int
end


"""
    Hash.init_lsh!(::LshSimHashParams, rng, IdType)

Initialize the LSH instance with SimHash. Ensures that all parameters
of the LSH and SimHash are properly glued together. Ensures that the number
of signatures produced by SimHash will match the number of hash tables in the LSH.
"""
function Hash.init_lsh!(
    sim_params::LshSimHashParams,
    rng::Rand,
    ::Type{Id},
)::LshSimHash where {Id,Rand<:AbstractRNG}
    lsh_params = sim_params.lsh_params
    hasher = initialize!(
        rng,
        lsh_params.n_tables * sim_params.signature_len,
        sim_params.sample_size,
        sim_params.vector_len,
    )

    Lsh(
        lsh_params.n_tables,
        lsh_params.n_buckets,
        lsh_params.max_bucket_len,
        SimhasherWrapper(UInt8(sim_params.signature_len), hasher),
        SubArray{Float},
        Id,
    )
end

end # LshSimHashWrapper
