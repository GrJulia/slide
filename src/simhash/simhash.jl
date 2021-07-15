module SimHash

export SimHasher, Signature, initialize!, signature

using Random: AbstractRNG, bitrand
using StatsBase: sample


const Signature = BitArray

struct SimHasher
    hashes::BitMatrix
    samples::Matrix{Int}

    SimHasher(hashes::BitMatrix, samples::Matrix{Int}) =
        size(hashes) == size(samples) ? new(hashes, samples) :
        error("Dimensions of matrices must match")
end

"""
    initialize!(r, n_hashes, sample_size, data_size)

Initialize SimHasher which can hash vector of dimension of `data_size`
into signature of `n_hashes` bits. Mutates the `r` argument.
"""
function initialize!(
    r::Rand,
    n_hashes::Int,
    sample_size::Int,
    data_size::Int,
)::SimHasher where {Rand<:AbstractRNG}
    @assert sample_size <= data_size "`sample_size` can't be larger than `data_size`"

    hashes = bitrand(r, (sample_size, n_hashes))
    samples = hcat([sample(1:data_size, sample_size, ordered = true) for _ = 1:n_hashes]...)

    SimHasher(hashes, samples)
end

select(x::T, flag::Bool) where {T<:Number} = flag ? x : -x

"""
    signature(sim_hasher, data)

Computes the signature of the data. Throws an error if the `data` size is incompatible with `SimHasher`.

The ith bit of the signature is computed as follows,
the `data` vector is permutated (defined by `simHasher.samples[:, i]`).
Then jth element of the permutated vector is multiplied by -1 if `sim_hasher.hashes[j, i]`
is equal to 0. Finally the sum of the transformed vector is sumed to produce
the ith bit of the signature.
"""
function signature(
    sim_hasher::SimHasher,
    data::A,
)::Signature where {A<:AbstractVector{<:Number}}
    sample_size, n_hashes = size(sim_hasher.hashes)
    @assert sample_size <= length(data) "`sample_size` can't be larger than `length(data_size)`"

    raw_signature = [
        sum(select.(getindex(data, sim_hasher.samples[:, i]), sim_hasher.hashes[:, i]))
        for i = 1:n_hashes
    ]

    map(x -> x >= 0, raw_signature)
end

end # SimHash
