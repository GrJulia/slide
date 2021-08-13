using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx

using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Hash: AbstractLshParams, init_lsh!
using Slide.Network.HashTables: SlideHashTables


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

Neuron(id::Id, batch_size::Int, input_dim::Int) = Neuron(
    id = id,
    weight = rand(Float, input_dim),
    bias = rand(Float),
    weight_gradients = zeros(Float, input_dim),
    bias_gradients = zeros(Float, batch_size),
    optimizer_attributes = AdamAttributes(input_dim),
    is_active = false,
)

@kwdef struct Layer{
    A<:AbstractLshParams,
    T<:AbstractOptimizerAttributes,
    F<:Function,
    Hasher<:AbstractHasher{SubArray{Float}},
}
    id::Id
    neurons::Vector{Neuron{T}}
    hash_tables::SlideHashTables{A,Hasher}
    layer_activation::F

    active_neurons::Vector{Vector{Id}}
    output::Vector{Vector{Float}}
end

struct SlideNetwork
    layers::Vector{Layer}
end

function Layer(
    id::Id,
    neurons::Vector{Neuron{T}},
    lsh_params::A,
    layer_activation::F;
    batch_size = 128,
) where {A<:AbstractLshParams,T<:AbstractOptimizerAttributes,F<:Function}

    hash_tables = SlideHashTables(
        lsh_params,
        extract_weights_and_ids(neurons),
    )

    undef_vec(U::DataType) = Vector{U}(undef, batch_size)

    Layer(
        id = id,
        neurons = neurons,
        hash_tables = hash_tables,
        layer_activation = layer_activation,
        active_neurons = undef_vec(Vector{Id}),
        output = undef_vec(Vector{Float}),
    )
end
