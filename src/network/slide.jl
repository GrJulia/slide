using Base: @kwdef
using Random: default_rng
using FLoops: ThreadedEx

using Slide.LSH: Lsh, AbstractHasher, add_batch!
using Slide.Hash: AbstractLshParams, init_lsh!


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
    grad_output_w::Vector
    grad_output_b::Vector
end

Neuron(id::Id, batch_size::Int, input_dim::Int) = Neuron(
    id = id,
    weight = rand(Float, input_dim),
    bias = rand(Float),
    weight_gradients = zeros(Float, input_dim),
    bias_gradients = zeros(Float, batch_size),
    optimizer_attributes = AdamAttributes(input_dim),
    is_active = false,
    grad_output_w = resize!(Vector{Array{Float}}(), batch_size),
    grad_output_b = resize!(Vector{Array{Float}}(), batch_size)
)

@kwdef mutable struct SlideHashTables{
    A<:AbstractLshParams,
    Hasher<:AbstractHasher{SubArray{Float}},
}
    lsh::SlideLsh{Hasher}
    lsh_params::A
    hashes::Matrix{Int}
    changed_ids::Set{Id}

    sampling_ratio::Int = 200
    min_threshold::Int = 90
end

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

    lsh = init_lsh!(lsh_params, default_rng(), Id)

    hashes = add_batch!(
        lsh,
        convert(
            Vector{Tuple{SubArray{Float},Id}},
            map(neuron -> (@view(neuron.weight[:]), neuron.id), neurons),
        );
        executor = ThreadedEx(),
    )

    hash_tables = SlideHashTables(
        lsh = lsh,
        lsh_params = lsh_params,
        hashes = hashes,
        changed_ids = Set{Id}(),
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



"""
    update!(hash_tables, neurons)
Recompute the hashes for the neurons but only for those that appear in the
`hash_tables.changed_ids`. Reinitialize `lsh`.
"""
function update!(
    hash_tables::SlideHashTables{A,Hasher},
    neurons::Vector{Neuron{T}},
) where {A<:AbstractLshParams,Hasher<:AbstractHasher{SubArray{Float}},T}
    lsh = init_lsh!(hash_tables.lsh_params, default_rng(), Id)

    # crashes anyway...
    changed_ids = [x for x in hash_tables.changed_ids]
    unchanged_neurons = filter(n -> !(n.id in changed_ids), neurons)
    unchanged_ids = map(n -> n.id, unchanged_neurons)

    not_changed_hashes = hash_tables.hashes[:, unchanged_ids]
    add_batch!(lsh, not_changed_hashes, unchanged_ids)

    new_hashes = add_batch!(
        lsh,
        convert(
            Vector{Tuple{SubArray{Float},Id}},
            map(id -> (@view(neurons[id].weight[:]), id), changed_ids),
        );
        executor = ThreadedEx(),
    )

    hash_tables.hashes[:, changed_ids] = new_hashes

    hash_tables.lsh = lsh
    hash_tables.changed_ids = Set{Id}()
end

"""
    Marks ids as changed.
"""
function mark_ids!(hash_tables::SlideHashTables{A,Hasher}, ids::Vector{Id}) where {A,Hasher}
    union!(hash_tables.changed_ids, ids)
end
