
mutable struct Neuron
    id::Id
    weight::Vector{Float32}
    bias::Float32
    active_inputs::Array{Id}
    activation_inputs::Array{Float}
    weight_gradients::Matrix{Float32}
    bias_gradients::Vector{Float32}
end

mutable struct Layer{F<:Function}
    id::Id
    neurons::Vector{Neuron}
    hash_table::HashTable
    layer_activation::F
end

mutable struct SlideNetwork
    layers::Vector{Layer}
end
