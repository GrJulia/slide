include("./lsh/Lsh.jl")
include("./simhash/Simhash.jl")
include("./dwta/DWTA.jl")

module Hash

export AbstractLshParams, LshParams, init_lsh, init_lsh!

using Base: @kwdef

abstract type AbstractLshParams end

"""
Default values are just for convenience.
"""
@kwdef struct LshParams <: AbstractLshParams
    n_buckets::Int = 10
    max_bucket_len::Int = 100
    n_tables::Int = 100
end

init_lsh() = error("unimplemented")
init_lsh!() = error("unimplemented")

end # Hash

include("./simhash_wrapper.jl")
include("./dwta_wrapper.jl")
