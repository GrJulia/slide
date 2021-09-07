using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx, @floop
using Distributions: Normal

using Slide: Float, Id, FloatVector
using Slide.Hash: AbstractLshParams, init_lsh!
using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Network.HashTables: SlideHashTables, update!
using Slide.Network.Optimizers: AbstractOptimizerAttributes


@kwdef mutable struct SlideLayer{
    A<:AbstractLshParams,
    F<:Function,
    Hasher<:AbstractHasher{FloatVector},
    Opt<:AbstractOptimizerAttributes,
} <: AbstractLayer
    id::Id
    biases::Vector{Float}
    weights::Matrix{Float}

    hash_tables::SlideHashTables{A,Hasher}
    activation::F

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
    stddev = 2 / sqrt(input_dim + output_dim)
    d = Normal(zero(Float), Float(stddev))

    weights = rand(d, input_dim, output_dim)
    hash_tables = SlideHashTables(lsh_params, extract_weights_and_ids(weights))

    SlideLayer(
        id = id,
        weights = weights,
        biases = rand(d, output_dim),
        hash_tables = hash_tables,
        activation = layer_activation,
        active_neuron_ids = Vector{Vector{Id}}(),
        output = Vector{Vector{Float}}(),
        bias_gradients = zeros(Float, output_dim, 1),
        weight_gradients = zeros(Float, input_dim, output_dim),
        is_neuron_active = zeros(Bool, output_dim),
        opt_attr = opt_attr,
    )
end

function to_inf(layer::SlideLayer)
    Dense(
        id = layer.id,
        biases = layer.biases,
        weights = layer.weights,
        activation = layer.activation,
        output = Vector{Vector{Float}}(),
        bias_gradients = zeros(Float, 1, 1),
        weight_gradients = zeros(Float, 1, 1),
        is_neuron_active = ones(Bool, 1),
        opt_attr = layer.opt_attr,
    )
end
