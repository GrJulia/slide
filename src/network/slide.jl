
mutable struct Neuron
    id::Id
    weight::Vector{Float32}
    bias::Float32
    active_inputs::Array{Id}
    activation_inputs::Array{Float}
    weight_gradients::Matrix{Float32}
    bias_gradients::Vector{Float32}
end

Neuron(id::Id, batch_size::Int, input_dim::Int) = Neuron(
    id,
    rand(input_dim) * 10,
    rand() * 10,
    zeros(batch_size),
    zeros(batch_size),
    zeros(input_dim, batch_size),
    zeros(batch_size),
)

abstract type AbstractOptimizerAttributes end

mutable struct AdamAttributes <: AbstractOptimizerAttributes
    m_dw::Vector{Float32}
    m_db::Float32
    v_dw::Vector{Float32}
    v_db::Float32
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
