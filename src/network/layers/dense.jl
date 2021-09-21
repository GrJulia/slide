using Base: @kwdef
using Distributions: Normal

using Slide: Float, Id, FloatVector
using Slide.Network.Optimizers: AbstractOptimizerAttributes, AdamAttributes

@kwdef mutable struct Dense{F<:Function,Opt<:AbstractOptimizerAttributes} <: AbstractLayer
    bias::Vector{Float}
    weights::Matrix{Float}

    activation::F
    output::Matrix{Float}

    bias_gradients::Matrix{Float}
    weight_gradients::Matrix{Float}

    is_neuron_active::Vector{Bool}
    opt_attr::Opt
end

function Dense(input_dim::Int, output_dim::Int, layer_activation::F) where {F}
    Dense(input_dim, output_dim, layer_activation, AdamAttributes(input_dim, output_dim))
end

function Dense(
    input_dim::Int,
    output_dim::Int,
    layer_activation::F,
    opt_attr::Opt,
) where {F,Opt}
    stddev = 2 / sqrt(input_dim + output_dim)
    d = Normal(zero(Float), Float(stddev))

    bias = rand(d, output_dim)
    weights = rand(d, input_dim, output_dim)

    Dense(weights, bias, layer_activation, opt_attr)
end

function Dense(weights::Matrix{Float}, bias::Vector{Float}, layer_activation::F) where {F}
    Dense(weights, bias, layer_activation, AdamAttributes(input_dim, output_dim))
end

function Dense(
    weights::Matrix{Float},
    bias::Vector{Float},
    layer_activation::F,
    opt_attr::Opt,
) where {F,Opt}
    input_dim, output_dim = size(weights)

    Dense(
        bias = bias,
        weights = weights,
        activation = layer_activation,
        output = Matrix{Float}(undef, 1, 1),
        bias_gradients = zeros(Float, output_dim, 1),
        weight_gradients = zeros(Float, input_dim, output_dim),
        is_neuron_active = ones(Bool, output_dim),
        opt_attr = opt_attr,
    )
end

function inference_mode(layer::Dense)
    layer
end
