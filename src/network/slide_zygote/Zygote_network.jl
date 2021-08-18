module ZygoteNetwork


export layer_forward_and_backward,
    forward_single_sample_zygote,
    forward_zygote!,
    train_zygote!,
    backward_zygote!,
    handle_batch_backward_zygote!

using Slide: Float, Id

include("slide_zygote_training_loop.jl")
include("slide_zygote_forward.jl")
include("slide_zygote_backward.jl")

end
