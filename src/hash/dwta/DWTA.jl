module DWTA

using Random: shuffle, rand, AbstractRNG, randperm
using IterTools

const Idx = UInt16
const BinId = UInt16

const Signatures = Vector{Idx}
const EMPTY_SAMPLING = zero(Idx)
const EMPTY_SAMPLING_VAL = Float32(-Inf)
const ZERO_VAL = zero(Float32)


struct DWTAHasher
    index_to_bin_ids::Vector{BinId}
    n_bins_per_idx_offsets::Vector{UInt16}
    n_hashes::Int
    log_n_hashes::Int32
    n_bins::Int
end

function initialize!(
    rng::Rand,
    n_hashes::Int,
    n_bins::Int,
    k::Int,
    data_len::Int,
)::DWTAHasher where {Rand<:AbstractRNG}
    perms = hcat((randperm(rng, data_len) for _ = 1:n_hashes)...)
    sampled_indices = @view perms[1:k, :]

    index_to_bin_ids = [Vector{BinId}() for _ = 1:data_len]

    # To each idx assign list of bins which include this idx
    for bin_id = 1:n_hashes
        @views for i in sampled_indices[:, bin_id]
            push!(index_to_bin_ids[i], bin_id)
        end
    end

    idx_to_n_bins = [length(bin_ids) for bin_ids in index_to_bin_ids]
    n_bins_per_idx_offsets = accumulate(+, idx_to_n_bins)
    pushfirst!(n_bins_per_idx_offsets, 0)

    index_to_bin_ids = collect(Iterators.flatten(index_to_bin_ids))

    log_n_hashes = ceil(Int32, log2(n_hashes))

    DWTAHasher(index_to_bin_ids, n_bins_per_idx_offsets, n_hashes, log_n_hashes, n_bins)
end

"""
Method for hashing a pair of integers which aims to avoid modular arithmetic. (TODO remove '% dwta.n_hashes' in case K * L = 2^M)
It computes f(x) = (a*x mod 2^w) div 2^(w-M) by doing (a*x) >> (w-M), where w in number of bits of the integer (32 in this case)
In other words, hash is computed by deriving M highest bits.
Link: https://en.wikipedia.org/wiki/Universal_hashing
"""
function two_universal_hash(dwta::DWTAHasher, bin_id::Int32, cnt::Int32)::Int32
    pair_hash = (bin_id << 6) + cnt
    return ((13557786907 * pair_hash) >>> (32 - dwta.log_n_hashes)) % dwta.n_hashes + 1
end

function signatures(
    dwta::DWTAHasher,
    data::A,
    wta::Bool;
    max_n_attemps = 100,
)::Signatures where {A<:AbstractVector{<:Number}}
    n_hashes, n_bins = dwta.n_hashes, dwta.n_bins
    hashes = fill(EMPTY_SAMPLING, n_hashes)
    max_vals_in_bins = fill(EMPTY_SAMPLING_VAL, n_hashes)
    bin_cnt = fill(one(Idx), n_bins)

    for i in eachindex(data)
        idx_start, idx_end =
            dwta.n_bins_per_idx_offsets[i] + 1, dwta.n_bins_per_idx_offsets[i+1]
        if idx_end - idx_start == -1 # 'idx' isn't present in any of the bins
            continue
        end

        val = data[i]
        # iterate over all bins which include index $i
        for bin_id in dwta.index_to_bin_ids[idx_start:idx_end]
            if val > max_vals_in_bins[bin_id] && val != ZERO_VAL
                max_vals_in_bins[bin_id] = val
                hashes[bin_id] = bin_cnt[bin_id]
            end
            bin_cnt[bin_id] += 1
        end
    end

    if wta
        return hashes
    end

    out_hashes = fill(EMPTY_SAMPLING, n_hashes)
    for bin_id = one(Int32):n_hashes
        curr_id, cnt = bin_id, zero(Int32)
        while hashes[curr_idx] == EMPTY_SAMPLING
            cnt += 1
            curr_id = two_universal_hash(dwta, bin_id, cnt)
            if cnt > min(n_hashes, max_n_attemps)
                break
            end
        end
        out_hashes[bin_id] = hashes[curr_id]
    end

    out_hashes
end

end # DWTA
