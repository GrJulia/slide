using LinearAlgebra: dot
using Statistics: mean, quantile
using Random: shuffle

using Slide: Float, Id, FloatVector

const ActiveMode = Val{:ActiveMode}
const RandomMode = Val{:RandomMode}
const MaxProductMode = Val{:MaxProductMode}


function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::ActiveMode,
) where {K<:FloatVector}
    (weights[:, active_neuron_ids]' * query) |> mean
end

function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::RandomMode,
) where {K<:FloatVector}
    n_active_neurons, n_neurons = length(active_neuron_ids), size(weights)[2]
    rand_neuron_ids = shuffle(1:n_neurons)[1:n_active_neurons]
    compute_avg_dot_product(query, weights, rand_neuron_ids, ActiveMode())
end

function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    mode::MaxProductMode,
) where {K<:FloatVector}
    n_active_neurons = length(active_neuron_ids)
    all_dot_products = weights' * query
    all_dot_products[partialsortperm(all_dot_products, 1:n_active_neurons, rev = true)] |>
    mean
end

function precision_at_k(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
) where {K<:FloatVector}
    top_active_products_ratio, relevant_products_ratio = [1], [0.03]
    logs = []
    for (top_active, relevance) in zip(top_active_products_ratio, relevant_products_ratio)
        # Find the largest dot products
        n_active_neurons = length(active_neuron_ids)
        n_top_active_neurons_used = floor(Int, top_active * n_active_neurons)

        active_neuron_dot_products =
            sort(weights[:, active_neuron_ids]' * query, rev = true)
        top_dot_products = active_neuron_dot_products[1:n_top_active_neurons_used]

        # Compute precision for the most relevant neurons
        all_dot_products = sort(weights' * query, rev = true)
        idx = floor(Int, relevance * length(all_dot_products))
        relevance_threshold = all_dot_products[idx]

        n_top_products = length(top_dot_products)
        top_dot_products = reverse(top_dot_products)

        n_relevant_products =
            n_top_products - searchsortedfirst(top_dot_products, relevance_threshold) + 1
        precision = n_relevant_products / n_top_products

        push!(logs, ("precision at top $(top_active) for relevance=$(relevance)", precision))
    end
    return logs
end

function log_dot_product_metrics(
    id::Id,
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
) where {K<:FloatVector}
    modes = ["ActiveMode", "RandomMode", "MaxProductMode"]
    compute_metric(mode) =
        compute_avg_dot_product(query, weights, active_neuron_ids, Val{Symbol(mode)}())
    res = [(mode, compute_metric(mode)) for mode in modes]

    precision = precision_at_k(query, weights, active_neuron_ids)

    for (key, val) in cat(res, precision, dims=1)
        @info "$(key)_$id" val
    end
end
