include("./lsh/Lsh.jl")
include("./simhash/Simhash.jl")

module Hash

export LshParams, init_lsh, init_lsh!

using Base: @kwdef

"""
Default values are just for convience.
"""
@kwdef struct LshParams
    n_buckets::Int = 10
    max_bucket_len::Int = 100
    n_tables::Int = 100
end

init_lsh() = error("unimplemented")
init_lsh!() = error("unimplemented")

end # Hash

include("./simhash_wrapper.jl")
