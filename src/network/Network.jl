module Network

const Float = typeof(1.0)

export Neuron,
    OptimizerAttributes,
    OptimizerNeuron,
    AdamAttributes,
    build_activated_neurons_single_sample,
    forward_single_sample,
    Layer,
    SlideNetwork,
    HashTable,
    build_network,
    forward!,
    batch_input,
    one_hot,
    backward!,
    update_weight!,
    numerical_gradient_weights,
    numerical_gradient_bias,
    empty_neurons_attributes!,
    build_and_train,
    Float,
    Optimizer,
    AdamOptimizer,
    optimizer_step,
    gradient,
    relu,
    get_deterministic_hash,
    handle_batch,
    handle_batch_backward,
    sparse_logit_cross_entropy

include("vanilla_hash_table.jl")
include("activations.jl")
include("slide.jl")
include("utils.jl")
include("optimizer.jl")
include("slide_forward.jl")
include("slide_backward.jl")
include("training_loop.jl")

end
