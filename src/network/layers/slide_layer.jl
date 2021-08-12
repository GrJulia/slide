using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx, @floop

using Slide: Float, Id
using Slide.Hash: AbstractLshParams, init_lsh!
using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Network.HashTables: SlideHashTables


@kwdef struct SlideLayer{
    A<:AbstractLshParams,
    T<:AbstractOptimizerAttributes,
    F<:Function,
    Hasher<:AbstractHasher{SubArray{Float}},
} <: AbstractLayer
    id::Id
    neurons::Vector{Neuron{T}}
    hash_tables::SlideHashTables{A,Hasher}
    layer_activation::F

    active_neurons::Vector{Vector{Id}}
    output::Vector{Vector{Float}}
end

function SlideLayer(
    id::Id,
    neurons::Vector{Neuron{T}},
    lsh_params::A,
    layer_activation::F,
) where {A<:AbstractLshParams,T<:AbstractOptimizerAttributes,F<:Function}

    lsh = init_lsh!(lsh_params, default_rng(), Id)

    hashes = add_batch!(
        lsh,
        convert(
            Vector{Tuple{SubArray{Float},Id}},
            map(neuron -> (@view(neuron.weight[:]), neuron.id), neurons),
        );
        executor = ThreadedEx(),
    )

    hash_tables = SlideHashTables(lsh_params, convert_neurons_to_batch(neurons))

    SlideLayer(
        id = id,
        neurons = neurons,
        hash_tables = hash_tables,
        layer_activation = layer_activation,
        active_neurons = Vector{Vector{Id}}(),
        output = Vector{Vector{Float}}(),
    )
end

new_batch!(::AbstractLayer, ::Int) = nothing

function new_batch!(
    layer::SlideLayer{A,T,F,H},
    batch_size::Int;
    executor = ThreadedEx(),
) where {A,T,F,H}
    resize!(layer.active_neurons, batch_size)
    resize!(layer.output, batch_size)

    @floop executor for neuron in layer.neurons
        reset!(neuron, batch_size)
    end
end
