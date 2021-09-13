module LshDwtaWrapper

export LshDwtaParams, get_dwta_params

using Base.Iterators: partition
using Random: AbstractRNG

using Slide: Float, FloatVector
using Slide.LSH: AbstractHasher, Lsh
using Slide.DWTA: DwtaHasher, initialize!, signature
using Slide.Hash: LshParams, AbstractLshParams

import Slide.Hash
import Slide.LSH


struct DwtaHasherWrapper <: AbstractHasher{FloatVector}
    hasher::DwtaHasher
    n_tables::UInt8
    n_bins::UInt8
    log_bin_size::UInt8
    densification::Bool
end

@inline function compute_signature_of_k_bins(
    bin_hashes::A,
    log_bin_size::UInt8,
)::Int where {T<:Number,A<:AbstractVector{T}}
    map(enumerate(bin_hashes)) do (i, h)
        (h - 1) << ((i - 1) * log_bin_size)
    end |> sum
end

function LSH.compute_signatures!(
    signatures::T,
    h::DwtaHasherWrapper,
    elem::K,
) where {T<:AbstractArray{Int},K<:FloatVector}
    raw_signature = signature(h.hasher, elem, h.densification)
    raw_signature_chunks = partition(raw_signature, h.n_bins)

    @inbounds for (i, bin_hashes) in enumerate(raw_signature_chunks)
        signatures[i] = compute_signature_of_k_bins(bin_hashes, h.log_bin_size)
    end
end

function LSH.compute_signatures(
    h::DwtaHasherWrapper,
    elem::K,
)::Vector{Int} where {K<:FloatVector}
    signatures = Vector{Int}(undef, h.n_tables)

    LSH.compute_signatures!(signatures, h, elem)

    signatures
end

@inline function LSH.compute_query_signatures(
    h::DwtaHasherWrapper,
    elem::K,
)::Vector{Int} where {K<:FloatVector}
    LSH.compute_signatures(h, elem)
end

@inline function LSH.compute_query_signatures!(
    signatures::T,
    h::DwtaHasherWrapper,
    elem::K,
) where {T<:AbstractArray{Int},K<:FloatVector}
    LSH.compute_signatures!(signatures, h, elem)
end

const LshDwta{Id} = Lsh{FloatVector,Id,DwtaHasherWrapper}

struct LshDwtaParams <: AbstractLshParams
    lsh_params::LshParams
    n_bins::UInt8
    n_indices_per_bin::UInt8
    data_len::Int
    densification::Bool
end

function Hash.init_lsh!(
    dwta_params::LshDwtaParams,
    rng::Rand,
    ::Type{Id},
)::LshDwta where {Id,Rand<:AbstractRNG}
    lsh_params = dwta_params.lsh_params
    hasher = initialize!(
        rng,
        UInt32(lsh_params.n_tables * dwta_params.n_bins),
        UInt32(dwta_params.n_indices_per_bin),
        UInt32(dwta_params.data_len),
    )

    log_bin_size = floor(UInt8, log2(dwta_params.n_indices_per_bin))

    Lsh(
        lsh_params.n_tables,
        lsh_params.n_buckets,
        lsh_params.max_bucket_len,
        DwtaHasherWrapper(
            hasher,
            lsh_params.n_tables,
            dwta_params.n_bins,
            log_bin_size,
            dwta_params.densification,
        ),
        AbstractVector{Float},
        Id,
    )
end

function get_dwta_params(
    params::LshParams,
    layer_sizes::Vector{Int};
    input_size::Int,
    n_bins::UInt8,
    n_indices_per_bin::UInt8,
    densification::Bool = false,
)::Vector{LshDwtaParams}
    lsh_params = Vector{LshDwtaParams}()

    prev_n_neurons = input_size
    for n_neurons in layer_sizes
        dwtaparams =
            LshDwtaParams(params, n_bins, n_indices_per_bin, prev_n_neurons, densification)
        push!(lsh_params, dwtaparams)
        prev_n_neurons = n_neurons
    end

    lsh_params
end

end # LshDWTAWrapper
