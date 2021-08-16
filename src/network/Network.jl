module Network


export AbstractScheduler,
    AdamAttributes,
    AdamOptimizer,
    Batch,
    Optimizer,
    SlideNetwork,
    backward!,
    batch_input,
    build_network,
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
    reset!,
    sparse_logit_cross_entropy,
    train!,
    update!,
    update_weight!,
    layer_forward_and_backward,
    forward_single_sample_zygote,
    forward_zygote!,
    train_zygote!,
    backward_zygote!,
    handle_batch_backward_zygote!


using Slide.LSH: Lsh
using Slide: Float, Id

include("hashtables.jl")
include("layers/Layers.jl")

include("scheduler.jl")
include("activations.jl")
include("slide.jl")
include("utils.jl")
include("optimizer.jl")
include("slide_forward.jl")
include("slide_backward.jl")
include("training_loop.jl")
include("slide_zygote/slide_zygote_training_loop.jl")
include("slide_zygote/slide_zygote_forward.jl")
include("slide_zygote/slide_zygote_backward.jl")

end
