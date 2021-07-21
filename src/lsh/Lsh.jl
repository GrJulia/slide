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
abstract type AbstractHasher{K} end

function compute_signatures(h::AbstractHasher{K}, elem::K)::Vector{Int} where {K}
    error("unimplemented")
end

function compute_query_signatures(h::AbstractHasher{K}, elem::K)::Vector{Int} where {K}
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
) where {K, V,Hasher<:AbstractHasher{K}} =
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
    compute_signaturess = compute_signatures(lsh.hash, key)

    for (compute_signatures, ht) in zip(compute_signaturess, lsh.hash_tables)
        add!(ht, compute_signatures, elem)
    end
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
    retrieve(lsh, elem)

From each table similar elements to the `elem` are retrieved.
"""
function retrieve(lsh::Lsh{K,V,Hasher}, key::K)::Set{V} where {K,V,Hasher<:AbstractHasher{K}}
    signatures = compute_query_signatures(lsh.hash, key)

    # Possible inconsistency in the SLIDE paper: Union vs Intersecion.
    reduce(
        zip(signatures, lsh.hash_tables),
        init = V[],
    ) do acc::Vector{V}, (signature, ht)::Tuple{Int,HashTable{V}}
        retrieved = retrieve(ht, signature)
        vcat(acc, retrieved)
    end |> Set{V}
end

end # LSH
