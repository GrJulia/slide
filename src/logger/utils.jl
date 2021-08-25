using LinearAlgebra: dot
using Statistics: mean, quantile
using Random: shuffle

using Slide: Float, Id, FloatVector
using Slide.SlideLogger: log_scalar!


function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::Val{:active_neurons},
) where {K<:FloatVector}
    active_neurons_weights = @view(weights[:, active_neuron_ids])
    mapslices(w -> dot(query, w), active_neurons_weights, dims = (1)) |> mean
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

function precision_at_k(
    id::Id,
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
) where {K<:FloatVector}
    top_ks, targets = [0.2, 0.2], [0.2, 0.1]

    active_neurons_weights = @view(weights[:, active_neuron_ids])
    active_neurons_dot_products = sort(
        reshape(
            mapslices(w -> dot(query, w), active_neurons_weights, dims = (1)),
            (length(active_neuron_ids)),
        ),
    )
    top_k_neurons =
        [floor(Int, (1 - top_k) * length(active_neurons_dot_products)) for top_k in top_ks]

    all_dot_products =
        reshape(mapslices(w -> dot(query, w), weights, dims = (1)), (size(weights)[2]))

    quantiles = quantile(all_dot_products, [1 - target for target in targets])

    top_k_active_neurons_dot_products =
        [active_neurons_dot_products[top_k:end] for top_k in top_k_neurons]
    precisions = [
        (length(top_k_products) - searchsortedfirst(top_k_products, quantile)) /
        length(top_k_products) for
        (top_k_products, quantile) in zip(top_k_active_neurons_dot_products, quantiles)
    ]


    for (precision, top_k, target) in zip(precisions, top_ks, targets)
        @info "precision at top $(top_k) for top $(target) products_$id" precision
    end
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
        @info "$(key)_$id" val
    end

    precision_at_k(id, query, weights, active_neuron_ids)
end
