include("./lsh/Lsh.jl")
include("./simhash/Simhash.jl")

module Hash

export LshParams, init_lsh, init_lsh!

struct LshParams
    n_buckets::Int
    max_bucket_len::Int
    n_tables::Int
end

init_lsh() = error("unimplemented")
init_lsh!() = error("unimplemented")

end # Hash

include("./simhash_wrapper.jl")
