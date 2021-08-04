module DWTA

# TODO add export

using Random: shuffle, rand, AbstractRNG, randperm
using IterTools

const Idx = UInt8

const Signatures = Vector{Idx}
const EMPTY_SAMPLING = zero(Idx)
const EMPTY_SAMPLING_VAL = Float32(-Inf)
const ZERO_VAL = zero(Float32)
const MAX_N_ATTEMPS = UInt32(100)


struct DWTAHasher
    indices_in_bin::Matrix{Int32}
    n_hashes::UInt32
    log_n_hashes::UInt32
end

function initialize!(
    rng::Rand,
    n_hashes::UInt32,
    k::UInt32,
    data_len::UInt32,
)::DWTAHasher where {Rand<:AbstractRNG}
    n_perms = ceil(UInt32, n_hashes * k / data_len)
    perms = vcat((randperm(rng, data_len) for _ = 1:n_perms)...)
    indices_in_bin = reshape(perms[1:n_hashes*k], (k, n_hashes))
    
    log_n_hashes = ceil(UInt32, log2(n_hashes))

    DWTAHasher(indices_in_bin, n_hashes, log_n_hashes)
end

"""
Method for hashing a pair of integers which aims to avoid modular arithmetic. (TODO remove '% dwta.n_hashes' in case K * L = 2^M)
It computes f(x) = (a*x mod 2^w) div 2^(w-M) by doing (a*x) >> (w-M), where w in number of bits of the integer (32 in this case)
In other words, hash is computed by deriving M highest bits.
Link: https://en.wikipedia.org/wiki/Universal_hashing
"""
function two_universal_hash(dwta::DWTAHasher, bin_idx::UInt32, cnt::UInt32)::UInt32
    pair_hash = (bin_idx << 6) + cnt
    return ((13557786907 * pair_hash) >>> (32 - dwta.log_n_hashes)) % dwta.n_hashes + 1
end

function signature(
    dwta::DWTAHasher,
    data::A,
    densification::Bool,
)::Signatures where {A<:AbstractVector{<:Number}}
    indices_in_bin = dwta.indices_in_bin
    n_hashes = dwta.n_hashes
    hashes = fill!(Signatures(undef, n_hashes), EMPTY_SAMPLING)

    for i = 1:n_hashes
        hashes[i] = argmax(view(data, indices_in_bin[:, i]))
        if data[indices_in_bin[hashes[i], i]] == ZERO_VAL
            hashes[i] = EMPTY_SAMPLING
        end
    end

    if !densification
        return hashes
    end

    out_hashes = Signatures(undef, n_hashes)
    for bin_idx = one(UInt32):n_hashes
        curr_idx, cnt = bin_idx, zero(UInt32)
        while hashes[curr_idx] == EMPTY_SAMPLING
            cnt += one(UInt32)
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
