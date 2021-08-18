using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx, @floop

using Slide: Float, Id, LshBatch
using Slide.Hash: AbstractLshParams, init_lsh!
using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Network.HashTables: SlideHashTables
using Slide.Network.Optimizers: AbstractOptimizerAttributes


@kwdef mutable struct SlideLayer{
    A<:AbstractLshParams,
    F<:Function,
    Hasher<:AbstractHasher{SubArray{Float}},
    Opt<:AbstractOptimizerAttributes,
} <: AbstractLayer
    id::Id
    biases::Vector{Float}
    weights::Matrix{Float}

    hash_tables::SlideHashTables{A,Hasher}
    layer_activation::F

    active_neuron_ids::Vector{Vector{Id}}
    output::Vector{Vector{Float}}

    bias_gradients::Matrix{Float}
    weight_gradients::Matrix{Float}
    is_neuron_active::Vector{Bool}

    opt_attr::Opt
end

function SlideLayer(
    id::Id,
    input_dim::Int,
    output_dim::Int,
    lsh_params::A,
    layer_activation::F,
    opt_attr::Opt,
) where {A<:AbstractLshParams,F<:Function,Opt<:AbstractOptimizerAttributes}
    weights = rand(Float, input_dim, output_dim)
    hash_tables = SlideHashTables(lsh_params, extract_weights_and_ids(weights))

    SlideLayer(
        id = id,
        weights = weights,
        biases = rand(Float, output_dim),
        hash_tables = hash_tables,
        layer_activation = layer_activation,
        active_neuron_ids = Vector{Vector{Id}}(),
        output = Vector{Vector{Float}}(),
        bias_gradients = zeros(Float, output_dim, 1),
        weight_gradients = zeros(Float, input_dim, output_dim),
        is_neuron_active = zeros(Bool, output_dim),
        opt_attr = opt_attr,
    )
end

new_batch!(::AbstractLayer, ::Int) = nothing

function new_batch!(layer::SlideLayer{A,F,H,O}, batch_size::Int) where {A,F,H,O}
    resize!(layer.active_neuron_ids, batch_size)
    resize!(layer.output, batch_size)
    fill!(layer.is_neuron_active, 0)
end

function zero_grads!(layer::SlideLayer{A,F,H,O}, batch_size::Int) where {A,F,H,O}
    fill!(layer.weight_gradients, 0)
    layer.bias_gradients = zeros(Float, length(layer.biases), batch_size)
end

function extract_weights_and_ids(weights::A)::LshBatch where {A<:AbstractMatrix{Float}}
    convert(LshBatch, map(i -> (@view(weights[:, i]), i), 1:size(weights, 2)))
end
