module Layers

export AbstractLayer,
    SlideLayer, new_batch!, extract_weights_and_ids, forward_single_sample!, prep_backprop!

abstract type AbstractLayer end

include("slide_layer.jl")
include("forward.jl")

end
