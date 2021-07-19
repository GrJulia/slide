
mutable struct Neuron
    id::Id
    weight::Vector{Float32}
    bias::Float64
    activation_input::Array{Int64}
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
