module Layers

export AbstractLayer,
    Dense,
    SlideLayer,
    new_batch!,
    extract_weights_and_ids,
    forward_single_sample!,
    prep_backprop!,
    calculate_error!,
    calculate_wgrads!,
    update_htable!,
    reinit_htable!,
    inference_mode

using Slide: Float, Id

abstract type AbstractLayer end

update_htable!(::AbstractLayer) = nothing
reinit_htable!(::AbstractLayer) = nothing

const SlideOutput = Tuple{Vector{Float},Vector{Id}}

include("dense.jl")
include("slide_layer.jl")
include("utils.jl")
include("dense_forward.jl")
include("slide_forward.jl")
include("backward.jl")
include("conversion.jl")

end
