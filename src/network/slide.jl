using JLD2

using Slide.Network: Layers
using Slide.Network.Layers: AbstractLayer

struct SlideNetwork
    layers::Vector{<:AbstractLayer}
end

SlideNetwork(layers...) = SlideNetwork(collect(layers))

function new_batch!(network::SlideNetwork, batch_size::Int)
    for layer in network.layers
        Layers.new_batch!(layer, batch_size)
    end
end

function zero_grads!(network::SlideNetwork, batch_size::Int)
    for layer in network.layers
        Layers.zero_grads!(layer, batch_size)
    end
end

function save(network::SlideNetwork, model_path::String)
    JLD2.jldsave(joinpath(model_path, "checkpoint.jld2"); network)
end

function load(model_path::String)
    JLD2.load(joinpath(model_path, "checkpoint.jld2"))

function inference_mode(network::SlideNetwork)
    SlideNetwork(map(Layers.inference_mode, network.layers))
end
