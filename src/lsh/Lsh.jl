module LSH

export AbstractHasher, Lsh, add!, retrieve

using DataStructures: CircularBuffer

""" AbstractHasher Interface/Trait """

"""
Supertype for hashers which can hash elements of type T.

Subtypes of the `AbstractHasher{T}` should implement following methods:
- `compute_signatures(h::AbstractHasher{T}, elem::T)::Vector{Int}`
- `compute_query_signatures(h::AbstractHasher{T}, elem::T)::Vector{Int}`

Most of the implementations can have `compute_query_signatures` being equal to `compute_signatures`,
the distinction is only important for the asymmetric-LSH implementations.
"""
abstract type AbstractHasher{T} end

function compute_signatures(h::AbstractHasher{T}, elem::T)::Vector{Int} where {T}
    error("unimplemented")
end

function compute_query_signatures(h::AbstractHasher{T}, elem::T)::Vector{Int} where {T}
    error("unimplemented")
end


""" LSH implementation """

const Bucket{T} = CircularBuffer{T}

struct HashTable{T}
    buckets::Vector{Bucket{T}}
end

struct Lsh{T,Hasher<:AbstractHasher{T}}
    hash::Hasher
    hash_tables::Vector{HashTable{T}}
end


HashTable(max_len::Int, n_buckets::Int, ::Type{T}) where {T} =
    HashTable{T}([Bucket{T}(max_len) for _ = 1:n_buckets])

Lsh(
    n_tables::Int,
    n_buckets::Int,
    max_len::Int,
    hasher::Hasher,
    ::Type{T},
) where {T,Hasher<:AbstractHasher{T}} =
    Lsh{T,Hasher}(hasher, [HashTable(max_len, n_buckets, T) for _ = 1:n_tables])


@inline function compute_bucket_for_signature(x::Int, max_value::Int)::Int
    (x % max_value) + 1
end

"""
    add!(table, signature, elem)

Adds `elem` to the bucket pointed by `signature`. If the selected bucket contains
more elements than the capacity of the bucket after addition then the oldest element
is removed from the table (FIFO).
"""
function add!(table::HashTable{T}, signature::Int, elem::T) where {T}
    bucket_id = compute_bucket_for_signature(signature, length(table.buckets))

    push!(table.buckets[bucket_id], elem)
end

"""
    add!(lsh, elem)

Computes the signatures of the `elem` and for each pair of table & signature
insert element into the table to the bucket selected by signature.
"""
function add!(lsh::Lsh{T,Hasher}, elem::T) where {T,Hasher<:AbstractHasher{T}}
    compute_signaturess = compute_signatures(lsh.hash, elem)

    for (compute_signatures, ht) in zip(compute_signaturess, lsh.hash_tables)
        add!(ht, compute_signatures, elem)
    end
end

"""
    retrieve(table, signature)

Returns contents of the bucket selected by `signature`.
"""
function retrieve(table::HashTable{T}, signature::Int)::Bucket{T} where {T}
    bucket_id = compute_bucket_for_signature(signature, length(table.buckets))

    table.buckets[bucket_id]
end

"""
    retrieve(lsh, elem)

From each table similar elements to the `elem` are retrieved.
"""
function retrieve(lsh::Lsh{T,Hasher}, elem::T)::Set{T} where {T,Hasher<:AbstractHasher{T}}
    signatures = compute_query_signatures(lsh.hash, elem)

    function _helper(acc, (signature, hash_table))
        retrieved = retrieve(hash_table, signature)
        vcat(acc, retrieved)
    end

    reduce(_helper, zip(signatures, lsh.hash_tables), init = []) |> Set
end

end # LSH
