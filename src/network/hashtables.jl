module HashTables

export SlideHashTables, update!

using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx

using Slide: Float, Id, LshBatch, FloatVector
using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Hash: AbstractLshParams, init_lsh!


const SlideLsh{Hasher} = Lsh{FloatVector,Id,Hasher}

@kwdef mutable struct SlideHashTables{
    A<:AbstractLshParams,
    Hasher<:AbstractHasher{FloatVector},
}
    lsh::SlideLsh{Hasher}
    lsh_params::A

    sampling_ratio::Float = Float(1 / 200)
    min_threshold::Int = 90
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
`neurons` is a vector of pairs of `(id, weight)`. Reinitializes `lsh`.
"""
function update!(
    hash_tables::SlideHashTables{A,Hasher},
    neurons::LshBatch,
) where {A<:AbstractLshParams,Hasher<:AbstractHasher{FloatVector}}
    hash_tables.lsh = init_and_populate_lsh(hash_tables.lsh_params, neurons)
end

end # HashTables
