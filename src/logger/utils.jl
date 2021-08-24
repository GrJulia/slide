using LinearAlgebra: dot
using Statistics: mean
using Random: shuffle

using Slide: Float, Id, FloatVector
using Slide.SlideLogger: log_scalar!


function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::Val{:active_neurons},
) where {K<:FloatVector}
    active_weights = @view(weights[:, active_neuron_ids])
    mapslices(w -> dot(query, w), active_weights, dims = (1)) |> mean
end

function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::Val{:random_neurons},
) where {K<:FloatVector}
    n_active_neurons, n_neurons = length(active_neuron_ids), size(weights)[2]
    rand_neuron_ids = shuffle(1:n_neurons)[1:n_active_neurons]
    compute_avg_dot_product(query, weights, rand_neuron_ids, Val{:active_neurons}())
end

function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::Val{:max_product},
) where {K<:FloatVector}
    n_active_neurons = length(active_neuron_ids)
    all_dot_products =
        reshape(mapslices(w -> dot(query, w), weights, dims = (1)), (size(weights)[2]))
    all_dot_products[partialsortperm(all_dot_products, 1:n_active_neurons, rev = true)] |>
    mean
end

function log_dot_product_metrics(
    id::Id,
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
) where {K<:FloatVector}
    res = [
        (
            mode,
            compute_avg_dot_product(query, weights, active_neuron_ids, Val{Symbol(mode)}()),
        ) for mode in ["active_neurons", "random_neurons", "max_product"]
    ]
    for (key, val) in res
        @info key * "_$id" val
    end
end
