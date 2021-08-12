using Base: @kwdef

using Slide: Float, Id, LshBatch

abstract type AbstractOptimizerAttributes end

mutable struct AdamAttributes <: AbstractOptimizerAttributes
    m_dw::Vector{Float}
    m_db::Float
    v_dw::Vector{Float}
    v_db::Float
end

AdamAttributes(input_dim::Int) = AdamAttributes(zeros(input_dim), 0, zeros(input_dim), 0)

@kwdef mutable struct Neuron{W<:AbstractOptimizerAttributes}
    id::Id
    weight::Vector{Float}
    bias::Float
    weight_gradients::Vector{Float}
    bias_gradients::Vector{Float}
    optimizer_attributes::W
    is_active::Bool
end

Neuron(id::Id, input_dim::Int) = Neuron(
    id = id,
    weight = rand(Float, input_dim),
    bias = rand(Float),
    weight_gradients = zeros(Float, input_dim),
    bias_gradients = Vector{Float}(),
    optimizer_attributes = AdamAttributes(input_dim),
    is_active = false,
)

function reset!(neuron::Neuron{T}, batch_size::Int) where {T}
    resize!(neuron.bias_gradients, batch_size)

    neuron.weight_gradients =
        fill!(neuron.weight_gradients, zero(eltype(neuron.weight_gradients)))
    neuron.bias_gradients =
        fill!(neuron.bias_gradients, zero(eltype(neuron.bias_gradients)))
    neuron.is_active = false
end

function extract_weights_and_ids(
    neurons::A,
)::LshBatch where {T,A<:AbstractVector{Neuron{T}}}
    convert(LshBatch, map(neuron -> (@view(neuron.weight[:]), neuron.id), neurons))
end
