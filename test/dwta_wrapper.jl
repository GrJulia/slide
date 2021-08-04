using Random: default_rng

using Slide: Float
using Slide.Hash: LshParams, init_lsh!
using Slide.LshDWTAWrapper: LshDWTAParams
using Slide.LSH: compute_signatures

lsh_params = LshParams(n_buckets = 10, max_bucket_len = 10, n_tables = 2)
dwta_params = LshDWTAParams(lsh_params, 6, 8, 10)
lsh_with_dwta = init_lsh!(dwta_params, default_rng(), Int)

x = Vector{Float}([0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.4, 0.3, 0.2, 0.1])
println(compute_signatures(lsh_with_dwta.hash, (@view x[:])))
