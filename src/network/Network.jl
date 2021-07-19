module Network

export Neuron,
    Layer,
    SlideNetwork,
    HashTable,
    build_network,
    forward,
    batch_input,
    one_hot,
    cross_entropy,
    backward!,
    update_weight!,
    numerical_gradient,
    empty_neurons_attributes!,
    build_and_train

include("training_loop.jl")
include("utils.jl")
include("vanilla_hash_table.jl")
include("activations.jl")
include("slide.jl")
include("slide_forward.jl")
include("slide_backward.jl")

end
