module DWTA

using Random: shuffle, rand, AbstractRNG, randperm
using IterTools

const IdxType = UInt16
const Signatures = Vector{IdxType}
const EMPTY_SAMPLING = zero(IdxType)
const EMPTY_SAMPLING_VAL = Float32(-Inf)
const ZERO_ELEM = zero(Float32)
const MAX_N_ATTEMPS = IdxType(100)


struct DWTAHasher
    idx_to_bins::Vector{IdxType}
    n_bins_per_idx_offsets::Vector{IdxType}
    n_hashes::IdxType
    log_n_hashes::Int32
    n_bins::IdxType
end

function initialize!(
    rng::Rand,
    n_hashes::Int,
    n_bins::Int,
    k::Int,
    data_len::Int,
)::DWTAHasher where {Rand<:AbstractRNG}
    temps = repeat(1:data_len, outer = (1, n_hashes)) # Vector of shape (data_len, n_hashes) s.t. temps[:, i] = 1:data_len
    perms = mapslices(col -> shuffle(rng, col), temps, dims = [1])
    sampled_indices = @view perms[1:k, :]

    idxs_to_bins = [Vector{IdxType}() for _ = 1:data_len]

    # To each idx assign list of bins including this idx
    for bin_idx = 1:n_hashes
        @views for idx in sampled_indices[:, bin_idx]
            push!(idxs_to_bins[idx], bin_idx)
        end
    end

    n_bins_per_idx = [length(idx_to_bins) for idx_to_bins in idxs_to_bins]
    n_bins_per_idx_offsets = accumulate(+, n_bins_per_idx)
    pushfirst!(n_bins_per_idx_offsets, 0)

    flat_idxs_to_bins = collect(Iterators.flatten(idxs_to_bins))

    log_n_hashes = ceil(Int32, log2(n_hashes))

    DWTAHasher(flat_idxs_to_bins, n_bins_per_idx_offsets, n_hashes, log_n_hashes, n_bins)
end

"""
Method for hashing a pair of integers which aims to avoid modular arithmetic. (TODO remove '% dwta.n_hashes' in case K * L = 2^M)
It computes f(x) = (a*x mod 2^w) div 2^(w-M) by doing (a*x) >> (w-M), where w in number of bits of the integer (32 in this case)
In other words, hash is computed by deriving M highest bits.
Link: https://en.wikipedia.org/wiki/Universal_hashing
"""
function two_universal_hash(dwta::DWTAHasher, bin_idx::Int32, cnt::Int32)::Int32
    pair_hash = (bin_idx << 6) + cnt
    return ((13557786906 * pair_hash) >> (32 - dwta.log_n_hashes)) % dwta.n_hashes + 1
end

function signatures(
    dwta::DWTAHasher,
    data::A,
    wta::Bool,
)::Signatures where {A<:AbstractVector{<:Number}}
    n_hashes, n_bins = dwta.n_hashes, dwta.n_bins
    hashes = fill(EMPTY_SAMPLING, n_hashes)
    max_vals_in_bins = fill(EMPTY_SAMPLING_VAL, n_hashes)
    bin_cnt = fill(one(IdxType), n_bins)

    for i in eachindex(data)
        idx_start, idx_end = dwta.n_bins_per_idx_offsets[i] + 1, dwta.n_bins_per_idx_offsets[i+1]
        if idx_end - idx_start == -1 # 'idx' isn't present in any of the bins
            continue
        end

        val = data[i]
        for bin_idx in dwta.idx_to_bins[idx_start:idx_end]
            if val > max_vals_in_bins[bin_idx] && val != ZERO_ELEM
                max_vals_in_bins[bin_idx] = val
                hashes[bin_idx] = bin_cnt[bin_idx]
            end
            bin_cnt[bin_idx] += 1
        end
    end

    if wta
        return hashes
    end

    out_hashes = fill(EMPTY_SAMPLING, n_hashes)
    for bin_idx = one(Int32):n_hashes
        curr_idx, cnt = bin_idx, zero(Int32)
        while hashes[curr_idx] == EMPTY_SAMPLING
            cnt += 1
            curr_idx = two_universal_hash(dwta, bin_idx, cnt)
            if cnt > min(n_hashes, MAX_N_ATTEMPS)
                break
            end
        end
        out_hashes[bin_idx] = hashes[curr_idx]
    end

    out_hashes
end

end # DWTA
