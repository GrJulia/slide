module Network

export Neuron, Layer, SlideNetwork, HashTable, build_network, forward, batch_input

include("vanilla_hash_table.jl")
include("activations.jl")
include("slide.jl")
include("slide_forward.jl")

end
