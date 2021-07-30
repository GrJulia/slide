
abstract type AbstractOptimizerAttributes end

mutable struct AdamAttributes <: AbstractOptimizerAttributes
    m_dw::Vector{Float}
    m_db::Float
    v_dw::Vector{Float}
    v_db::Float
end

AdamAttributes(input_dim::Int) = AdamAttributes(zeros(input_dim), 0, zeros(input_dim), 0)

mutable struct Neuron{W<:AbstractOptimizerAttributes}
    id::Id
    weight::Vector{Float}
    bias::Float
    active_inputs::Array{Id}
    activation_inputs::Array{Float}
    pre_activation_inputs::Array{Float}
    weight_gradients::Matrix{Float}
    bias_gradients::Vector{Float}
    optimizer_attributes::W
end

Neuron(id::Id, batch_size::Int, input_dim::Int) = Neuron(
    id,
    rand(Float, input_dim),
    rand(Float),
    zeros(Id, batch_size),
    zeros(Float, batch_size),
    zeros(Float, batch_size),
    zeros(Float, input_dim, batch_size),
    zeros(Float, batch_size),
    AdamAttributes(input_dim),
)

struct Layer{T<:AbstractOptimizerAttributes,F<:Function}
    id::Id
    neurons::Vector{Neuron{T}}
    hash_table::HashTable
    layer_activation::F
end

struct SlideNetwork
    layers::Vector{Layer}
end
