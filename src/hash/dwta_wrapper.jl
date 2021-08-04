module LshDWTAWrapper

export LshDWTAParams, get_dwta_params

using Base.Iterators: partition
using Random: AbstractRNG

using Slide: Float
using Slide.LSH: AbstractHasher, Lsh
using Slide.DWTA: DWTAHasher, initialize!, signature
using Slide.Hash: LshParams, AbstractLshParams

import Slide.Hash
import Slide.LSH


struct DWTAHasherWrapper <: AbstractHasher{SubArray{Float}}
    hasher::DWTAHasher
    n_tables::UInt8
    n_bins::UInt8
    log_n_indices_per_bin::UInt8
end

@inline function compute_signature_from_bin_hashes(
    bin_hashes::A,
    log_n_indices_per_bin::UInt8,
)::Int where {T<:Number,A<:AbstractVector{T}}
        map(enumerate(bin_hashes)) do (i, h)
            (h-1) << ((i-1)*log_n_indices_per_bin)
        end |> sum
end

function LSH.compute_signatures!(
    signatures::T,
    h::DWTAHasherWrapper,
    elem::SubArray{Float},
) where {T<:AbstractArray{Int}}
    raw_signature = signature(h.hasher, elem, false)
    raw_signature_chunks = partition(raw_signature, h.n_bins)

    @inbounds for (i, bin_hashes) in enumerate(raw_signature_chunks)
        signatures[i] = compute_signature_from_bin_hashes(bin_hashes, h.log_n_indices_per_bin)
    end
end

function LSH.compute_signatures(
    h::DWTAHasherWrapper,
    elem::SubArray{Float},
)::Vector{Int}
    signatures = Vector{Int}(undef, h.n_tables)

    LSH.compute_signatures!(signatures, h, elem)

    signatures
end

@inline function LSH.compute_query_signatures(
    h::DWTAHasherWrapper,
    elem::SubArray{Float},
)::Vector{Int}
    LSH.compute_signatures(h, elem)
end

@inline function LSH.compute_query_signatures!(
    signatures::T,
    h::DWTAHasherWrapper,
    elem::SubArray{Float},
) where {T<:AbstractArray{Int}}
    LSH.compute_signatures!(signatures, h, elem)
end

const LshDWTA{Id} = Lsh{SubArray{Float},Id,DWTAHasherWrapper}

struct LshDWTAParams <: AbstractLshParams
    lsh_params::LshParams
    n_bins::UInt8
    n_indices_per_bin::UInt8
    data_len::Int
end

function Hash.init_lsh!(
    dwta_params::LshDWTAParams,
    rng::Rand,
    ::Type{Id},
)::LshDWTA where {Id,Rand<:AbstractRNG}
    lsh_params = dwta_params.lsh_params
    hasher = initialize!(
        rng,
        UInt32(lsh_params.n_tables * dwta_params.n_bins),
        UInt32(dwta_params.n_indices_per_bin),
        UInt32(dwta_params.data_len),
    )

    log_n_indices_per_bin = floor(UInt8, log2(dwta_params.n_indices_per_bin))
    # @assert log2(lsh_params.n_buckets) == dwta_params.n_bins * log_n_indices_per_bin

    Lsh(
        lsh_params.n_tables,
        lsh_params.n_buckets,
        lsh_params.max_bucket_len,
        DWTAHasherWrapper(hasher, lsh_params.n_tables, dwta_params.n_bins, log_n_indices_per_bin),
        SubArray{Float},
        Id,
    )
end

function get_dwta_params(
    params::LshParams,
    layer_sizes::Vector{Int};
    input_size::Int,
    n_bins::UInt8,
    n_indices_per_bin::UInt8,
)::Vector{LshDWTAParams}
    lsh_params = Vector{LshDWTAParams}()

    prev_n_neurons = input_size
    for n_neurons in layer_sizes
        dwtaparams = LshDWTAParams(
            params,
            n_bins,
            n_indices_per_bin,
            prev_n_neurons,
        )
        push!(lsh_params, dwtaparams)
        prev_n_neurons = n_neurons
    end

lsh_params
end

end # LshDWTAWrapper
