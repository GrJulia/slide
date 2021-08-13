module Network


export AbstractScheduler,
    AdamAttributes,
    AdamOptimizer,
    Batch,
    Layer,
    Neuron,
    Optimizer,
    OptimizerAttributes,
    SlideNetwork,
    backward!,
    batch_input,
    build_network,
    extract_weights_and_ids,
    forward!,
    forward_single_sample,
    gradient,
    handle_batch_backward,
    negative_sparse_logit_cross_entropy,
    numerical_gradient_bias,
    numerical_gradient_weights,
    one_hot,
    optimizer_end_epoch_step!,
    optimizer_step!,
    PeriodicScheduler,
    predict_class,
    relu,
    sparse_logit_cross_entropy,
    train!,
    update!,
    update_weight!,
    zero_neuron_attributes!


using Slide.LSH: Lsh
using Slide: Float, Id

include("hashtables.jl")
include("scheduler.jl")
include("activations.jl")
include("slide.jl")
include("utils.jl")
include("optimizer.jl")
include("slide_forward.jl")
include("slide_backward.jl")
include("training_loop.jl")

end
