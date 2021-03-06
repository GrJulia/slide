module HashTables

export SlideHashTables, update!, reinit!

using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx
using LinearAlgebra: norm

using Slide: Float, Id, LshBatch, FloatVector
using Slide.LSH: Lsh, AbstractHasher, add_batch!, reset!
using Slide.Hash: AbstractLshParams, init_lsh!
using Slide.LshAsymmetricHasher: LshAsymHasherParams, update_norm!


const SlideLsh{Hasher} = Lsh{FloatVector,Id,Hasher}

@kwdef mutable struct SlideHashTables{
    A<:AbstractLshParams,
    Hasher<:AbstractHasher{FloatVector},
}
    lsh::SlideLsh{Hasher}
    lsh_params::A

    sampling_ratio::Float = Float(0.1)
    min_threshold::Int = 90
end

@inline function init_and_populate_lsh(
    lsh_params::LshAsymHasherParams,
    neurons::LshBatch,
)::SlideLsh{<:AbstractHasher{FloatVector}}
    lsh_params.max_norm = maximum(map(weights -> norm(weights), first.(neurons)))

    lsh = init_lsh!(lsh_params, default_rng(), Id)
    add_batch!(lsh, neurons; executor = ThreadedEx())

    lsh
end

@inline function init_and_populate_lsh(
    lsh_params::A,
    neurons::LshBatch,
)::SlideLsh{<:AbstractHasher{FloatVector}} where {A<:AbstractLshParams}
    lsh = init_lsh!(lsh_params, default_rng(), Id)
    add_batch!(lsh, neurons; executor = ThreadedEx())

    lsh
end

function SlideHashTables(
    lsh_params::A,
    neurons::LshBatch,
)::SlideHashTables{A,<:AbstractHasher{FloatVector}} where {A<:AbstractLshParams}
    SlideHashTables(
        lsh = init_and_populate_lsh(lsh_params, neurons),
        lsh_params = lsh_params,
    )
end

"""
    update!(hash_tables, neurons)

Recompute the hashtables for the `neurons`.
`neurons` is a vector of pairs of `(id, weight)`.
"""
function update!(
    hash_tables::SlideHashTables{A,Hasher},
    neurons::LshBatch,
) where {A<:AbstractLshParams,Hasher<:AbstractHasher{FloatVector}}
    reset!(hash_tables.lsh)
    add_batch!(hash_tables.lsh, neurons; executor = ThreadedEx())
end

"""
update!(hash_tables, neurons)

Recompute the hashtables for the `neurons`.
Update norm value in the asymmetric lsh hasher.
`neurons` is a vector of pairs of `(id, weight)`.
"""
function update!(
    hash_tables::SlideHashTables{LshAsymHasherParams,Hasher},
    neurons::LshBatch,
) where {Hasher<:AbstractHasher{FloatVector}}
    new_norm = maximum(map(weights -> norm(weights), first.(neurons)))
    reset!(hash_tables.lsh)

    update_norm!(hash_tables.lsh.hash, new_norm)
    add_batch!(hash_tables.lsh, neurons; executor = ThreadedEx())
end

"""
    re_init!(hash_tables, neurons)

Recompute the hashtables for the `neurons`.
`neurons` is a vector of pairs of `(id, weight)`.
Reinitializes `lsh` including creating new random hasher.
"""
function reinit!(
    hash_tables::SlideHashTables{A,Hasher},
    neurons::LshBatch,
) where {A<:AbstractLshParams,Hasher<:AbstractHasher{FloatVector}}
    hash_tables.lsh = init_and_populate_lsh(hash_tables.lsh_params, neurons)
end

end # HashTables
