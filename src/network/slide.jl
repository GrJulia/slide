using Slide.Network: Layers
using Slide.Network.Layers: AbstractLayer

struct SlideNetwork
    layers::Vector{<:AbstractLayer}
end

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
