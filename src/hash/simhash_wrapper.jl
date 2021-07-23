
module LshSimHashWrapper

export LshSimHashParams

using Base.Iterators: partition
using Random: AbstractRNG

using ..LSH: AbstractHasher, Lsh, add!, retrieve
using ..SimHash: SimHasher, Signature, initialize!, signature
using ..Hash: LshParams

import ..Hash
import ..LSH


struct SimhasherWrapper{A<:AbstractVector{<:Number}} <: AbstractHasher{A}
    signature_len::UInt8
    hasher::SimHasher

    function SimhasherWrapper{A}(
        signature_len::UInt8,
        hasher::SimHasher,
    ) where {A<:AbstractVector{<:Number}}
        signature_len >= 8 * sizeof(Int) && error(
            "Signature length can't be greater than $(8 * sizeof(Int)), got $signature_len",
        )
        new(signature_len, hasher)
    end
end


function LSH.compute_signatures(
    h::SimhasherWrapper{A},
    elem::A,
)::Vector{Int} where {A<:AbstractVector{<:Number}}
    raw_signature = signature(h.hasher, elem)
    raw_signature_chunks = partition(raw_signature, h.signature_len)
    map(raw_signature_chunks) do boolarray
        BitArray(boolarray).chunks[1]
    end
end

@inline function LSH.compute_query_signatures(
    h::SimhasherWrapper{A},
    elem::A,
)::Vector{Int} where {A<:AbstractVector{<:Number}}
    LSH.compute_signatures(h, elem)
end


const LshSimHash{Id} = Lsh{Vector{Float32},Id,SimhasherWrapper{Vector{Float32}}}

struct LshSimHashParams
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
        SimhasherWrapper{Vector{Float32}}(UInt8(sim_params.signature_len), hasher),
        Vector{Float32},
        Id,
    )
end

end # LshSimHashWrapper