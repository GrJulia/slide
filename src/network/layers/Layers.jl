module Layers

export AbstractLayer,
    SlideLayer, new_batch!, Neuron, AdamAttributes, extract_weights_and_ids

abstract type AbstractLayer end

include("neuron.jl")
include("slide_layer.jl")

end
