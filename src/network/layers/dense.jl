using Base: @kwdef
using Distributions: Normal

using Slide: Float, Id, FloatVector
using Slide.Network.Optimizers: AbstractOptimizerAttributes

@kwdef mutable struct Dense{F<:Function,Opt<:AbstractOptimizerAttributes} <: AbstractLayer
    id::Id
    biases::Vector{Float}
    weights::Matrix{Float}

    activation::F
    output::Vector{Vector{Float}}

    bias_gradients::Matrix{Float}
    weight_gradients::Matrix{Float}

    is_neuron_active::Vector{Bool}
    opt_attr::Opt
end

function Dense(
    id::Id,
    input_dim::Int,
    output_dim::Int,
    layer_activation::F,
    opt_attr::Opt,
) where {F,Opt}
    stddev = 2 / sqrt(input_dim + output_dim)
    d = Normal(zero(Float), Float(stddev))

    Dense(
        id = id,
        biases = rand(d, output_dim),
        weights = rand(d, input_dim, output_dim),
        activation = layer_activation,
        output = Vector{Vector{Float}}(),
        bias_gradients = zeros(Float, output_dim, 1),
        weight_gradients = zeros(Float, input_dim, output_dim),
        is_neuron_active = ones(Bool, output_dim),
        opt_attr = opt_attr,
    )
end

function to_inf(layer::Dense)
    layer
end