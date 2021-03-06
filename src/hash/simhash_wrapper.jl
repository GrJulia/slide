module LshSimHashWrapper

export LshSimHashParams, get_simhash_params

using Base.Iterators: partition
using Random: AbstractRNG

using Slide: Float, FloatVector
using Slide.LSH: AbstractHasher, Lsh
using Slide.SimHash: SimHasher, Signature, initialize!, signature, signature_len
using Slide.Hash: LshParams, AbstractLshParams

import Slide.Hash
import Slide.LSH

struct SimhasherWrapper <: AbstractHasher{FloatVector}
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
    elem::K,
) where {T<:AbstractVector{Int},K<:FloatVector}
    raw_signature = signature(h.hasher, elem)
    raw_signature_chunks = partition(raw_signature, h.signature_len)

    @inbounds for (i, array) in enumerate(raw_signature_chunks)
        signatures[i] = compute_integer_from_bin(array)
    end
end

function LSH.compute_signatures(
    h::SimhasherWrapper,
    elem::K,
)::Vector{Int} where {K<:FloatVector}
    n_signatures = signature_len(h.hasher) ÷ h.signature_len
    signatures = Vector{Int}(undef, n_signatures)

    LSH.compute_signatures!(signatures, h, elem)

    signatures
end

@inline function LSH.compute_query_signatures(
    h::SimhasherWrapper,
    elem::K,
)::Vector{Int} where {K<:FloatVector}
    LSH.compute_signatures(h, elem)
end

@inline function LSH.compute_query_signatures!(
    signatures::T,
    h::SimhasherWrapper,
    elem::K,
) where {T<:AbstractVector{Int},K<:FloatVector}
    LSH.compute_signatures!(signatures, h, elem)
end


const LshSimHash{Id} = Lsh{FloatVector,Id,SimhasherWrapper}

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

function LSH.init_hasher(
    sim_params::LshSimHashParams,
    rng::Rand,
)::SimhasherWrapper where {Rand<:AbstractRNG}
    lsh_params = sim_params.lsh_params
    hasher = initialize!(
        rng,
        lsh_params.n_tables * sim_params.signature_len,
        sim_params.sample_size,
        sim_params.vector_len,
    )
    SimhasherWrapper(UInt8(sim_params.signature_len), hasher)
end

function Hash.init_lsh!(
    sim_params::LshSimHashParams,
    rng::Rand,
    ::Type{Id},
)::LshSimHash where {Id,Rand<:AbstractRNG}
    lsh_params = sim_params.lsh_params
    Lsh(
        lsh_params.n_tables,
        lsh_params.n_buckets,
        lsh_params.max_bucket_len,
        LSH.init_hasher(sim_params, rng),
        FloatVector,
        Id,
    )
end

function get_simhash_params(
    params::LshParams,
    layer_sizes::Vector{Int};
    input_size::Int,
    signature_len::Int,
    sample_ratio::Float,
)::Vector{LshSimHashParams}
    lsh_params = Vector{LshSimHashParams}()

    prev_n_neurons = input_size
    for n_neurons in layer_sizes
        simparams = LshSimHashParams(
            params,
            prev_n_neurons,
            signature_len,
            floor(Int, prev_n_neurons ÷ sample_ratio),
        )
        push!(lsh_params, simparams)
        prev_n_neurons = n_neurons
    end

    lsh_params
end

end # LshSimHashWrapper
