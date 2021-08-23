module LSH

export AbstractHasher, Lsh, add!, retrieve

using DataStructures: CircularBuffer
using FLoops: @floop, SequentialEx
using Random: AbstractRNG

using Slide: Float

""" AbstractHasher Interface/Trait """

"""
Supertype for hashers which can hash elements of type T.

Subtypes of the `AbstractHasher{T}` should implement following methods:
- `compute_signatures(h::AbstractHasher{T}, elem::T)::Vector{Int}`
- `compute_query_signatures(h::AbstractHasher{T}, elem::T)::Vector{Int}`
and in place version of those functions:
- `compute_signatures!(h::AbstractHasher{K}, elem::K, signature <: AbstractArray{Int})
- `compute_query_signatures!(h::AbstractHasher{K}, elem::K, signature <: AbstractArray{Int})`

Most of the implementations can have `compute_query_signatures` being equal to `compute_signatures`,
the distinction is only important for the asymmetric-LSH implementations.
"""
abstract type AbstractHasher{K} end

function compute_signatures(h::AbstractHasher{K}, elem::K)::Vector{Int} where {K}
    error("unimplemented")
end

function compute_signatures!(
    signatures::T,
    h::AbstractHasher{K},
    elem::K,
) where {K,T<:AbstractArray{Int}}
    error("unimplemented")
end

function compute_query_signatures(h::AbstractHasher{K}, elem::K)::Vector{Int} where {K}
    error("unimplemented")
end

function compute_query_signatures!(
    signature::T,
    h::AbstractHasher{K},
    elem::K,
) where {K,T<:AbstractArray{Int}}
    error("unimplemented")
end

function init_hasher(
    params,
    rng::Rand,
)::AbstractHasher where {Rand<:AbstractRNG} # TODO: move AbstractLshParams to Lsh?
    error("unimplemented")
end

""" LSH implementation """

const Bucket{V} = CircularBuffer{V}

struct HashTable{V}
    buckets::Vector{Bucket{V}}
end

struct Lsh{K,V,Hasher<:AbstractHasher{K}}
    hash::Hasher
    hash_tables::Vector{HashTable{V}}
end


HashTable(max_len::Int, n_buckets::Int, ::Type{V}) where {V} =
    HashTable{V}([Bucket{V}(max_len) for _ = 1:n_buckets])

Lsh(
    n_tables::Int,
    n_buckets::Int,
    max_len::Int,
    hasher::Hasher,
    ::Type{K},
    ::Type{V},
) where {K,V,Hasher<:AbstractHasher{K}} =
    Lsh{K,V,Hasher}(hasher, [HashTable(max_len, n_buckets, V) for _ = 1:n_tables])


@inline function compute_bucket_for_signature(x::Int, max_value::Int)::Int
    (x % max_value) + 1
end

"""
    add!(table, signature, elem)

Adds `elem` to the bucket pointed by `signature`. If the selected bucket contains
more elements than the capacity of the bucket after addition then the oldest element
is removed from the table (FIFO).
"""
function add!(table::HashTable{V}, signature::Int, elem::V) where {V}
    bucket_id = compute_bucket_for_signature(signature, length(table.buckets))

    push!(table.buckets[bucket_id], elem)
end

"""
    add!(lsh, elem)

Computes the signatures of the `elem` and for each pair of table & signature
insert element into the table to the bucket selected by signature.
"""
function add!(lsh::Lsh{K,V,Hasher}, key::K, elem::V) where {K,V,Hasher<:AbstractHasher{K}}
    signatures = compute_signatures(lsh.hash, key)
    for (signature, ht) in zip(signatures, lsh.hash_tables)
        add!(ht, signature, elem)
    end
end


"""
    add_batch!(lsh, signatures, elems)

For each element extract the signature from the signatures matrix and using it
insert its into the tables.
"""
function add_batch!(
    lsh::Lsh{K,V,Hasher},
    signatures::Matrix{Int},
    elems::T,
) where {K,V,Hasher<:AbstractHasher{K},T<:Vector{V}}
    @views @inbounds for (i, elem) in enumerate(elems)
        for (signature, ht) in zip(signatures[:, i], lsh.hash_tables)
            add!(ht, signature, elem)
        end
    end
end

"""
    add_batch!(lsh, batch, ex)

For each pair of (key, elem) in the batch computes the signatures and adds elem
to the tables. Returns matrix of the shape (length(tables), length(batch)).
Ith column of this matrix contains computed `signatures` of the ith element from the batch.
`ex` argument defines executing strategy of computation of signatures. Default is `SequentialEx`.
"""
function add_batch!(
    lsh::Lsh{K,V,Hasher},
    batch::Vector{Tuple{L,V}};
    executor = SequentialEx(),
)::Matrix{Int} where {K,L<:K,V,Hasher<:AbstractHasher{K}}
    n_tables = length(lsh.hash_tables)
    b_len = length(batch)

    signatures = Matrix{Int}(undef, n_tables, b_len)

    @views @inbounds @floop executor for i = 1:b_len
        key, _ = batch[i]
        compute_signatures!(signatures[:, i], lsh.hash, key)
    end

    add_batch!(lsh, signatures, map(((_, elem),) -> elem, batch))

    signatures
end

"""
    retrieve(table, signature)

Returns contents of the bucket selected by `signature`.
"""
function retrieve(table::HashTable{V}, signature::Int)::Bucket{V} where {V}
    bucket_id = compute_bucket_for_signature(signature, length(table.buckets))

    table.buckets[bucket_id]
end

"""
    retrieve(lsh, elem; threshold=nothing)

From each table similar elements to the `elem` are retrieved.
If threshold argument is not nothing then after the count of retrieved elements
will be higher than it no further element will be retrieved.

Possible inconsistency in the SLIDE paper: Union vs Intersecion.
"""
function retrieve(
    lsh::Lsh{K,V,Hasher},
    key::K;
    threshold::Union{Nothing,Int} = nothing,
)::Set{V} where {K,V,Hasher<:AbstractHasher{K}}
    signatures = compute_query_signatures(lsh.hash, key)

    similar_elems = Set{V}()

    for (signature, ht) in zip(signatures, lsh.hash_tables)
        retrieved = retrieve(ht, signature)
        union!(similar_elems, retrieved)

        if !isnothing(threshold) && length(similar_elems) >= threshold
            break
        end
    end

    similar_elems
end

end # LSH
