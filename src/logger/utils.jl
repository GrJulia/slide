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
    ::ActiveMode,
) where {K<:FloatVector}
    mean((weights[:, active_neuron_ids]' * query))
end

function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    ::RandomMode,
) where {K<:FloatVector}
    n_active_neurons, n_neurons = length(active_neuron_ids), size(weights)[2]
    rand_neuron_ids = shuffle(1:n_neurons)[1:n_active_neurons]
    compute_avg_dot_product(query, weights, rand_neuron_ids, ActiveMode())
end

function compute_avg_dot_product(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
    ::MaxProductMode,
) where {K<:FloatVector}
    n_active_neurons = length(active_neuron_ids)
    all_dot_products = weights' * query
    mean(partialsort(all_dot_products, 1:n_active_neurons, rev = true))
end

function precision_at_k(
    query::K,
    weights::Matrix{Float},
    active_neuron_ids::Vector{Id},
) where {K<:FloatVector}
    active_quantiles, all_quantiles = [1], [0.03]
    logs = []
    for (active_quantile, all_quantile) in zip(active_quantiles, all_quantiles)
        n_active_neurons, n_neurons = length(active_neuron_ids), size(weights, 2)
        scores = weights' * query

        # Find retrieved neurons with the largest scores
        n_largest_active_neurons = floor(Int, active_quantile * n_active_neurons)
        largest_active_neuron_scores =
            partialsort(scores[active_neuron_ids], 1:n_largest_active_neurons, rev = true)

        # Compute precision for these neurons
        sorted_scores = sort(scores, rev = true)
        n_largest_scores = floor(Int, all_quantile * n_neurons)
        relevance_threshold = sorted_scores[n_largest_scores]

        n_relevant_scores =
            searchsortedfirst(largest_active_neuron_scores, relevance_threshold, rev = true)
        precision = n_relevant_scores / n_largest_active_neurons

        push!(
            logs,
            (
                "precision at top $(active_quantile) for all_quantile=$(all_quantile)",
                precision,
            ),
        )
    end

    logs
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

    for (key, val) in cat(res, precision, dims = 1)
        @info "$(key)_$id" val
    end
end
