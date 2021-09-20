using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx, @floop
using Distributions: Normal

using Slide: Float, Id, FloatVector
using Slide.Hash: AbstractLshParams, init_lsh!
using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Network.HashTables: SlideHashTables, update!, reinit!
using Slide.Network.Optimizers: AbstractOptimizerAttributes, AdamAttributes


@kwdef mutable struct SlideLayer{
    A<:AbstractLshParams,
    F<:Function,
    Hasher<:AbstractHasher{FloatVector},
    Opt<:AbstractOptimizerAttributes,
} <: AbstractLayer
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
    input_dim::Int,
    output_dim::Int,
    lsh_params::A,
    layer_activation::F,
) where {F,A}
    SlideLayer(
        input_dim,
        output_dim,
        lsh_params,
        layer_activation,
        AdamAttributes(input_dim, output_dim),
    )
end

function SlideLayer(
    input_dim::Int,
    output_dim::Int,
    lsh_params::A,
    layer_activation::F,
    opt_attr::Opt,
) where {A<:AbstractLshParams,F<:Function,Opt<:AbstractOptimizerAttributes}
    stddev = 2 / sqrt(input_dim + output_dim)
    d = Normal(zero(Float), Float(stddev))

    weights = rand(d, input_dim, output_dim)
    biases = rand(d, output_dim)

    SlideLayer(weights, biases, layer_activation, lsh_params, opt_attr)
end


function SlideLayer(
    weights::Matrix{Float},
    bias::Vector{Float},
    layer_activation::F,
    lsh_params::A,
) where {F,A}
    SlideLayer(weights, bias, layer_activation, lsh_params, AdamAttributes(input_dim, output_dim))
end

function SlideLayer(
    weights::Matrix{Float},
    bias::Vector{Float},
    layer_activation::F,
    lsh_params::A,
    opt_attr::Opt,
) where {F,A,Opt}

    input_dim, output_dim = size(weights)
    hash_tables = SlideHashTables(lsh_params, extract_weights_and_ids(weights))

    SlideLayer(
        weights = weights,
        biases = bias,
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

function inference_mode(layer::SlideLayer)
    Dense(
        biases = layer.biases,
        weights = layer.weights,
        activation = layer.activation,
        output = Matrix{Float}(undef, 1, 1),
        bias_gradients = zeros(Float, 1, 1),
        weight_gradients = zeros(Float, 1, 1),
        is_neuron_active = ones(Bool, 1),
        opt_attr = layer.opt_attr,
    )
end
