using Slide.Network.Layers: AbstractLayer

struct SlideNetwork
    layers::Vector{<:AbstractLayer}
end
