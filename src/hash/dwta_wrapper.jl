using ..Hash: LshParams

struct LshDWTAParams
    lsh_params::LshParams
    vector_len::Int
    signature_len::Int
    top_k::Int
end
