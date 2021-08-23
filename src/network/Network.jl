module Network


export AbstractScheduler,
    Batch,
    SlideNetwork,
    backward!,
    batch_input,
    build_layer,
    build_network,
    compute_accuracy,
    forward!,
    forward_single_sample,
    gradient,
    handle_batch_backward,
    negative_sparse_logit_cross_entropy,
    numerical_gradient_bias,
    numerical_gradient_weights,
    one_hot,
    PeriodicScheduler,
    predict_class,
    relu,
    sparse_logit_cross_entropy,
    train!,
    update_weight!


using Slide.LSH: Lsh
using Slide: Float, Id

include("hashtables.jl")
include("activations.jl")

include("optimizers/Optimizers.jl")
include("layers/Layers.jl")

include("scheduler.jl")
include("slide.jl")
include("utils.jl")
include("slide_forward.jl")
include("slide_zygote/slide_zygote_backward.jl")
include("slide_backward.jl")
include("training_loop.jl")

end
