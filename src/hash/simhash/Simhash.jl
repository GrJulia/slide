module SimHash

export SimHasher, Signature, initialize!, signature, signature_len

using LinearAlgebra: dot
using Random: AbstractRNG, bitrand
using StatsBase: sample


const Signature = BitArray

struct SimHasher
    hashes::Matrix{Int8}
    samples::Matrix{Int}

    SimHasher(hashes::Matrix{Int8}, samples::Matrix{Int}) =
        size(hashes) == size(samples) ? new(hashes, samples) :
        error(
            "Expected that `size(samples)` should match `size(hashes)`. Got $(size(samples)) and $(size(hashes))",
        )
end


@inline signature_len(sim_hasher::SimHasher)::Int = size(sim_hasher.hashes)[2]

"""
    initialize!(r, n_hashes, subvector_length, data_length)

Initialize SimHasher which can hash vector of dimension of `data_length`
into signature of `n_hashes` bits. Mutates the `r` argument.
"""
function initialize!(
    rng::Rand,
    n_hashes::Int,
    subvector_length::Int,
    data_length::Int,
)::SimHasher where {Rand<:AbstractRNG}
    @assert subvector_length <= data_length "`subvector_length` can't be larger than `data_length`"

    hashes = sample(rng, Vector{Int8}([1, -1]), (subvector_length, n_hashes))
    samples = hcat(
        [sample(1:data_length, subvector_length, ordered = true) for _ = 1:n_hashes]...,
    )

    SimHasher(hashes, samples)
end


@inline function compute_hash(data, sampled_indices, hashes)
    subvector = getindex(data, sampled_indices)

    dot(subvector, hashes)
end

"""
    signature(sim_hasher, data)

Computes the signature of the data. Throws an error if the `data` size is incompatible with `SimHasher`.

The ith bit of the signature is computed as follows,
the `data` vector is sampled (defined by `sim_hasher.samples[:, i]`).
Then jth element of the subvector is multiplied by `sim_hasher.hashes[j, i]`.
If the simhash is initialized by the `initialize!` function the values
of `sim_hasher.hashes` are drawn from (1, -1) values. Finally the sum
of the transformed subvector is sumed to produce the ith bit of the signature.

See paragraph for SimHash from section 3.2 from the paper
`SLIDE: in Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems`.
"""
function signature(
    sim_hasher::SimHasher,
    data::A,
)::Signature where {A<:AbstractVector{<:Number}}
    subvector_length, n_hashes = size(sim_hasher.hashes)
    @assert subvector_length <= length(data) "`subvector_length` can't be larger than `length(data_length)`"

    raw_signature::Vector{Float32} = Vector{Float32}(undef, n_hashes)

    @inbounds for i = 1:n_hashes
        raw_signature[i] =
            compute_hash(data, sim_hasher.samples[:, i], sim_hasher.hashes[:, i])
    end

    map(x -> x >= 0, raw_signature)
end

end # SimHash
