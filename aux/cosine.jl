using Random: default_rng, shuffle
using Distributions: Normal
using LinearAlgebra: dot, norm
using Statistics: mean

using Slide: Float
using Slide.Hash: LshParams, init_lsh!
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.LshDwtaWrapper: LshDwtaParams
using Slide.LshAsymmetricHasher: LshAsymHasherParams
using Slide.LSH: compute_signatures, add!, add_batch!, retrieve
using Slide.SlideLogger: precision_at_k, compute_avg_dot_product


cosine_sim(x,y) = Float(dot(x,y)/(norm(x)*norm(y)))


function top_n_cosine_sim_weights(query, weights, n)
    cos_sims = []
    n_neurons = size(weights)[2]
    for i = 1:n_neurons
        cos_sim = cosine_sim(query, weights[:, i])
        push!(cos_sims, (cos_sim, i))
    end
    return sort(cos_sims, rev=true)[1:n]
end


lsh_params = LshParams(n_buckets = 1 << 9, max_bucket_len = 128, n_tables = 1)
vector_len = 128
simhash_params = LshSimHashParams(lsh_params, vector_len, 9, 3)
lsh_with_simhash = init_lsh!(simhash_params, default_rng(), Int)

d = Normal(zero(Float), 1)
@views begin
    n_neurons = 200000
    weights = Matrix{Float}(undef, vector_len, n_neurons)
    weights = rand(d, vector_len, n_neurons)
    neurons = [(weights[:, i], i) for i = 1:n_neurons]
end
add_batch!(lsh_with_simhash, neurons)


q = rand(d, vector_len)
ids_of_active_weights = collect(retrieve(lsh_with_simhash, q))
active_weights = weights[:, ids_of_active_weights]

cos_sims = []
for i = 1:size(active_weights)[end]
    push!(cos_sims, cosine_sim(q, active_weights[:, i]))
end

cos_sims = sort(cos_sims, rev=true)
all_best_cos_sims = top_n_cosine_sim_weights(q, weights, 5*length(cos_sims))
best_cos_sims = all_best_cos_sims[1:length(cos_sims)]

println("Cosine similarity of n retrieved weights: $(last.(cos_sims))\n")
println("Mean of cosine similarity of n retrieved weights: $(last.(cos_sims) |> mean)\n")
println("Mean of cosine similarity of n/5 top retrieved weights: $(last.(cos_sims[1:floor(Int, length(cos_sims)/5)]) |> mean)\n")

println("n top cosine similarities: $(first.(best_cos_sims))\n")
println("Mean of n top cosine similarities: $(first.(best_cos_sims) |> mean)\n")
println("Mean of 5n top cosine similarities: $(first.(all_best_cos_sims) |> mean)\n")
