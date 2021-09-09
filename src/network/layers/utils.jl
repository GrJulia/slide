
using Slide: Float, Id, FloatVector, LshBatch
"""
    Common layer utilities
"""

function extract_weights_and_ids(weights::A)::LshBatch where {A<:AbstractMatrix{Float}}
    convert(LshBatch, map(i -> (@view(weights[:, i]), i), 1:size(weights, 2)))
end

@inline function activation(layer::L) where {L<:AbstractLayer}
    layer.activation
end

@inline function get_output(layer::L, x_index::Int) where {L<:AbstractLayer}
    layer.active_neuron_ids[x_index], layer.output[x_index]
end

@inline function get_output(layer::L) where {L<:AbstractLayer}
    layer.active_neuron_ids, layer.output
end

@inline function get_bias_gradients(layer::L, x_index::Int) where {L<:AbstractLayer}
    @view layer.bias_gradients[:, x_index]
end

@inline function get_weights(layer::L) where {L<:AbstractLayer}
    layer.weights
end

@inline function get_weight_gradients(layer::L) where {L<:AbstractLayer}
    layer.weight_gradients
end

new_batch!(::AbstractLayer, ::Int) = nothing

function zero_grads!(layer::L, batch_size::Int) where {L<:AbstractLayer}
    fill!(layer.weight_gradients, 0)

    new_shape = (length(layer.biases), batch_size)

    if new_shape == size(layer.biases)
        fill!(layer.bias_gradients, 0)
    else
        layer.bias_gradients = zeros(Float, length(layer.biases), batch_size)
    end
end

const get_error = get_bias_gradients

@inline function set_error!(
    layer::L,
    x_index::Int,
    ids,
    error::T,
) where {L<:AbstractLayer,T<:FloatVector}
    get_error(layer, x_index)[ids] .= error
end

"""
    Slide layer specific utilities
"""
@inline function set_active!(
    layer::SlideLayer{A,F,H,O},
    active_ids::T,
) where {A,F,H,O,T<:AbstractVector{Id}}
    layer.is_neuron_active[active_ids] .= true
end

function new_batch!(layer::SlideLayer{A,F,H,O}, batch_size::Int) where {A,F,H,O}
    resize!(layer.active_neuron_ids, batch_size)
    resize!(layer.output, batch_size)
    fill!(layer.is_neuron_active, 0)
end

function update_htable!(layer::SlideLayer{A,F,H,O}) where {A,F,H,O}
    update!(layer.hash_tables, extract_weights_and_ids(layer.weights))
end

"""
    Dense layer specific utilities
"""
@inline function set_active!(::Dense{F,O}, ::Any) where {F,O}
    nothing
end

@inline function get_output(layer::Dense{F,O}, x_index::Int) where {F,O}
    1:length(layer.output[:, x_index]), @view layer.output[:, x_index]
end

@inline function get_output(layer::Dense{F,O}) where {F,O}
    layer.output
end

function new_batch!(layer::Dense{F,O}, batch_size::Int) where {F,O}
    layer.output = Matrix{Float}(undef, length(layer.biases), batch_size)
end
