
mutable struct Neuron
    id::Id
    weight::Vector{Float}
    bias::Float
    active_inputs::Array{Id}
    activation_inputs::Array{Float}
    weight_gradients::Matrix{Float}
    bias_gradients::Vector{Float}
end

Neuron(id::Id, batch_size::Int, input_dim::Int) = Neuron(
    id,
    rand(input_dim),
    rand(),
    zeros(Float, batch_size),
    zeros(Float, batch_size),
    zeros(Float, input_dim, batch_size),
    zeros(Float, batch_size),
)

abstract type AbstractOptimizerAttributes end

mutable struct AdamAttributes <: AbstractOptimizerAttributes
    m_dw::Vector{Float}
    m_db::Float
    v_dw::Vector{Float}
    v_db::Float
end

AdamAttributes(input_dim::Int) = AdamAttributes(zeros(input_dim), 0, zeros(input_dim), 0)

struct OptimizerNeuron{W<:AbstractOptimizerAttributes}
    neuron::Neuron
    optimizer_attributes::W
end

struct Layer{T<:AbstractOptimizerAttributes,F<:Function}
    id::Id
    neurons::Vector{OptimizerNeuron{T}}
    hash_table::HashTable
    layer_activation::F
end

struct SlideNetwork
    layers::Vector{Layer}
end
