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
    grad_output_w 
    grad_output_b
end

Neuron(id::Id, input_dim::Int) = Neuron(
    id = id,
    weight = rand(Float, input_dim),
    bias = rand(Float),
    weight_gradients = zeros(Float, input_dim),
    bias_gradients = Vector{Float}(),
    optimizer_attributes = AdamAttributes(input_dim),
    is_active = false,
    grad_output_w = Vector{Array{Float}}(),
    grad_output_b = Vector{Float}()
)

function reset!(neuron::Neuron{T}, batch_size::Int) where {T}
    resize!(neuron.grad_output_w, batch_size)
    resize!(neuron.grad_output_b, batch_size)
    neuron.grad_output_b =
        fill!(neuron.grad_output_b, zero(eltype(neuron.grad_output_b)))

    neuron.weight_gradients =
        fill!(neuron.weight_gradients, zero(eltype(neuron.weight_gradients)))
    resize!(neuron.bias_gradients, batch_size)
    neuron.bias_gradients =
        fill!(neuron.bias_gradients, zero(eltype(neuron.bias_gradients)))
    neuron.is_active = false
end

function extract_weights_and_ids(
    neurons::A,
)::LshBatch where {T,A<:AbstractVector{Neuron{T}}}
    convert(LshBatch, map(neuron -> (@view(neuron.weight[:]), neuron.id), neurons))
end
