using Slide: Float, Id, FloatVector
using Slide.Network: gradient


function backward_single_sample!(;
    layer::SlideLayer{A,F,H,O},
    grads::U,
    layer_input::T,
    x_index::Int,
) where {A,F,H,O,U<:FloatVector,T<:FloatVector}
    active_neuron_ids = layer.active_neuron_ids[x_index]
    _grad_calculations!(layer, layer_input, active_neuron_ids, grads, :, x_index, F)
end

function backward_single_sample_with_output!(;
    layer::SlideLayer{A,F,H,O},
    grads::U,
    layer_input::Tuple{T,P},
    output_grads::U,
    x_index::Int,
) where {A,F,H,O,U<:FloatVector,P<:AbstractVector{Id},T<:FloatVector}
    @views begin
        previous_output, previous_active_ids = layer_input
        active_neuron_ids = layer.active_neuron_ids[x_index]

        dz = _grad_calculations!(
            layer,
            previous_output,
            active_neuron_ids,
            grads,
            previous_active_ids,
            x_index,
            F,
        )

        previous_layer_output_len = length(previous_active_ids)
        output_grads[1:previous_layer_output_len] =
            layer.weights[previous_active_ids, active_neuron_ids] * dz
    end
end


@inline function _grad_calculations!(
    layer,
    previous_output,
    active_neuron_ids,
    loss_grad,
    previous_active_ids,
    x_index,
    F,
)
    @views begin
        layer.is_neuron_active[active_neuron_ids] .= true

        layer_output_len = length(layer.output[x_index])
        dz = loss_grad[1:layer_output_len] .* gradient(F, layer.output[x_index])
        layer.bias_gradients[active_neuron_ids, x_index] .= dz

        dz ./= length(layer.bias_gradients[1, :])
        for (k, neuron_id) in enumerate(active_neuron_ids)
            @. layer.weight_gradients[previous_active_ids, neuron_id] +=
                dz[k] * previous_output
        end

        dz
    end
end
