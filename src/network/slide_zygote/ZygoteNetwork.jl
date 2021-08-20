module ZygoteNetwork


export layer_forward_and_backward,
    train_zygote!, backward_zygote!, handle_batch_backward_zygote!

using Slide: Float, Id

include("slide_zygote_training_loop.jl")
include("slide_zygote_backward.jl")

end
