module LSH

export AbstractHasher, Lsh, add!, retrieve

const Bucket{T} = Vector{T}

""" AbstractHasher Interface/Trait """

"""
Supertype for hashers which can hash elements of type T.

Subtypes of the `AbstractHasher{T}` should implement following methods:
- `signature(h::AbstractHasher{T}, elem::T)::Vector{Int}`
- `query_signature(h::AbstractHasher{T}, elem::T)::Vector{Int}`

Most of the implementations can have `query_signature` being equal to `signature`,
the distinction is only important for the asymmetric-LSH implementations.
"""
abstract type AbstractHasher{T} end

function signature(h::AbstractHasher{T}, elem::T)::Vector{Int} where {T}
    error("unimplemented")
end

function query_signature(h::AbstractHasher{T}, elem::T)::Vector{Int} where {T}
    error("unimplemented")
end


""" LSH implementation """

const Bucket{T} = Vector{T}

struct HashTable{T}
    max_len::Int
    buckets::Vector{Bucket{T}}
end

struct Lsh{T,Hasher<:AbstractHasher{T}}
    hash::Hasher
    hash_tables::Vector{HashTable{T}}
end


HashTable(max_len::Int, n_buckets::Int, ::Type{T}) where {T} =
    HashTable{T}(max_len, [[] for _ = 1:n_buckets])

Lsh(
    n_tables::Int,
    n_buckets::Int,
    max_len::Int,
    hasher::Hasher,
    ::Type{T},
) where {T,Hasher<:AbstractHasher{T}} =
    Lsh{T,Hasher}(hasher, [HashTable(max_len, n_buckets, T) for _ = 1:n_tables])


@inline function number_to_index(x::Int, max_value::Int)::Int
    idx = x % max_value
    if idx == 0
        idx = max_value
    end

    idx
end

"""
    add!(table, signature, elem)

Adds `elem` to the bucket pointed by `signature`. If the selected bucket contains
more elements than `table.max_len` after addition then the oldest element is removed
from the table (FIFO).
"""
function add!(table::HashTable{T}, signature::Int, elem::T) where {T}
    bucket_id = number_to_index(signature, length(table.buckets))

    push!(table.buckets[bucket_id], elem)
    length(table.buckets[bucket_id]) > table.max_len && popfirst!(table.buckets[bucket_id])
end

"""
    add!(lsh, elem)

Computes the signatures of the `elem` and for each pair of table & signature
insert element into the table to the bucket selected by signature.
"""
function add!(lsh::Lsh{T,Hasher}, elem::T) where {T,Hasher<:AbstractHasher{T}}
    signatures = signature(lsh.hash, elem)

    for (signature, ht) in zip(signatures, lsh.hash_tables)
        add!(ht, signature, elem)
    end
end

"""
    retrieve(table, signature)

Returns contents of the bucket selected by `signature`.
"""
function retrieve(table::HashTable{T}, signature::Int)::Bucket{T} where {T}
    bucket_id = number_to_index(signature, length(table.buckets))

    table.buckets[bucket_id]
end

"""
    retrieve(lsh, elem)

From each table similar elements to the `elem` are retrieved.
"""
function retrieve(lsh::Lsh{T,Hasher}, elem::T)::Set{T} where {T,Hasher<:AbstractHasher{T}}
    signatures = query_signature(lsh.hash, elem)

    function _helper(acc, (signature, hash_table))
        retrieved = retrieve(hash_table, signature)
        vcat(acc, retrieved)
    end

    reduce(_helper, zip(signatures, lsh.hash_tables), init = []) |> Set
end

end # LSH
