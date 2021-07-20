
mutable struct Neuron
    id::Id
    weight::Vector{Float32}
    bias::Float32
    active_inputs::Array{Id}
    activation_inputs::Array{Float}
    weight_gradients::Matrix{Float32}
    bias_gradients::Vector{Float32}
end

abstract type OptimizerAttributes end

mutable struct AdamAttributes <: OptimizerAttributes
    m_dw::Vector{Float32}
    m_db::Float32
    v_dw::Vector{Float32}
    v_db::Float32
end

struct OptimizerNeuron{W<:OptimizerAttributes}
    neuron::Neuron
    optimizer_attributes::W
end

struct Layer{T, F<:Function}
    id::Id
    neurons::Vector{OptimizerNeuron{T}}
    hash_table::HashTable
    layer_activation::F
end

struct SlideNetwork
    layers::Vector{Layer}
end
