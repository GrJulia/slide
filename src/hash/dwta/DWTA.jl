module DWTA

using Random: shuffle, rand, AbstractRNG


const Signatures = Vector{Vector{UInt32}}

const EMPTY_SAMPLING = Float32(-Inf)

struct DWTAHasher
    idx_to_bins::Vector{Int}
    bins_offsets::Vector{Int}
    n_hashes::Int
    n_bins::Int
    log_n_hashes::Float32
    rand_hash::Int
end


function acc_prefixes(A::Vector{Int})
    function add_prefix(acc, x)
        if length(acc) > 0
            x += last(acc)
            push!(acc, x)
            acc
        else
            [x]
        end
    end
    reduce(add_prefix, A, init = [])
end

function initialize!(
    rng::Rand,
    n_tables::Int,
    n_bins::Int,
    k::Int,
    data_len::Int,
)::DWTAHasher where {Rand<:AbstractRNG}
    n_hashes = n_tables * n_bins
    temps = repeat(1:data_len, outer = (1, n_hashes)) # Vector of shape (data_len, n_hashes) s.t. temp[:, i] = 1:data_len
    perms = mapslices(row -> shuffle(rng, row), temps, dims = [1])
    bin_to_idxs = @view perms[1:k, :]

    idxs_to_bins = [Vector{Int}() for _ = 1:data_len]

    # To each idx assign list of bins including this idx
    for bin_idx = 1:n_hashes
        @views for idx in bin_to_idxs[:, bin_idx]
            push!(idxs_to_bins[idx], bin_idx)
        end
    end

    bins_offsets = [length(idx_to_bins) for idx_to_bins in idxs_to_bins]
    pushfirst!(bins_offsets, 0)
    bins_offsets = acc_prefixes(bins_offsets)

    idxs_to_bins = collect(Iterators.flatten(idxs_to_bins))

    log_n_hashes = Float32(log2(n_hashes))
    rand_hash = rand(Int32)
    if rand_hash % 2 == 0
        rand_hash += 1
    end

    DWTAHasher(idxs_to_bins, bins_offsets, n_hashes, n_bins, log_n_hashes, rand_hash)
end

function two_universal_hash(dwta::DWTAHasher, bin_idx::Int, cnt::Int)::Int
    tohash = ((bin_idx + 1) << 6) + cnt
    return (dwta.rand_hash * tohash << 3) >> (32 - dwta.log_n_hashes)
end

function signatures(
    dwta::DWTAHasher,
    data::A,
)::Signatures where {A<:AbstractVector{<:Number}}
    hashes = fill(EMPTY_SAMPLING, dwta.n_hashes)
    max_vals_in_bins = fill(EMPTY_SAMPLING, dwta.n_hashes)

    for idx in eachindex(data)
        idx_start, idx_end = dwta.bins_offsets[idx] + 1, dwta.bins_offsets[idx+1]
        if idx_end - idx_start == 0 # 'idx' isn't present in any of the bins
            continue
        end

        val = data[idx]
        bins_with_idx = @view dwta.idx_to_bins[idx_start:idx_end]
        for bin_idx in bins_with_idx
            if val > max_vals_in_bins[bin_idx]
                max_vals_in_bins[bin_idx] = val
                hashes[bin_idx] = idx
            end
        end
    end

    # Handling empty sampling (is it ever used?)
    out_hashes = fill(EMPTY_SAMPLING, dwta.n_hashes)
    for bin_idx in eachidex(hashes)
        curr_idx, cnt = bin_idx, 0
        while hashes[curr_idx] == EMPTY_SAMPLING
            cnt += 1
            curr_idx = two_universal_hash(dwta, bin_idx, cnt)
            if cnt > 100
                break
            end
        end
        out_hashes[bin_idx] = hashes[curr_idx]
    end

    @views [out_hashes[i:i+dwta.n_bins] for i = 1:n_bins:n_hashes-n_bins]
end

end # DWTA
