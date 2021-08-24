using LinearAlgebra: dot
using Statistics: mean

using Slide: FloatVector
using Slide.Logging: Logger, log_scalar!


function compute_avg_dot_product(query<:FloatVector, weights::Matrix{Float}, active_neuron_ids::Vector{Id}, mode::Val{:active_neurons})
    active_weights = @view(weights[:, active_neuron_ids])
    map(w -> dot(query, w), active_weights) |> mean
end

function compute_avg_dot_product(query<:FloatVector, weights::Matrix{Float}, active_neuron_ids::Vector{Id}, mode::Val{:random_neurons})
    n_active_neurons, n_neurons = length(active_neuron_ids), size(weights)[2]
    rand_neuron_ids = shuffle(1:n_neurons)[1:n_active_neurons]
    
    active_weights = @view(weights[:, rand_neuron_ids])
    map(w -> dot(query, w), active_weights) |> mean
end

function compute_avg_dot_product(query<:FloatVector, weights::Matrix{Float}, active_neuron_ids::Vector{Id}, mode::Val{:max_product})
    n_active_neurons= length(active_neuron_ids)
    all_dot_products = map(w -> dot(query, w), weights)
    all_dot_products[partialsortperm(all_dot_products, 1:n_active_neurons, rev=True)] |> mean
end

function log_dot_product_metrics(query<:FloatVector, weights::Matrix{Float}, active_neuron_ids::Vector{Id}, logger::Logger)
    res = [(mode, compute_avg_dot_product(query, weights, active_neuron_ids, Val{Symbol(mode)})) for mode in ["active_neurons", "random_neurons", "max_product"]]
    for (key, val) in res
        log_scalar!(logger, key, val)
    end
end
