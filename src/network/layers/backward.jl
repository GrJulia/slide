using LinearAlgebra: axpy!
using FLoops: @floop, ThreadedEx, SequentialEx

using Slide: Float, Id, FloatVector
using Slide.Network: gradient

"""
    Common layer backward functions
"""
function calculate_error!(
    layer::L,
    next_layer::K,
    x_index::Int,
) where {L<:AbstractLayer,K<:AbstractLayer}
    F = typeof(activation(layer))

    current_active_neuron_ids, output = get_output(layer, x_index)
    active_neuron_ids, _ = get_output(next_layer, x_index)

    @views begin
        grad = get_error(next_layer, x_index)[active_neuron_ids]
        weights = get_weights(next_layer)[current_active_neuron_ids, active_neuron_ids]

        set_error!(
            layer,
            x_index,
            current_active_neuron_ids,
            (weights * grad) .* gradient(F, output),
        )
    end

end

function _calculate_wgrads!(
    layer,
    input,
    previous_active_ids,
    x_index;
    executor = SequentialEx(),
)
    current_active_neuron_ids, _ = get_output(layer, x_index)
    set_active!(layer, current_active_neuron_ids)

    @views begin
        dz = get_error(layer, x_index)
        dWs = get_weight_gradients(layer)[previous_active_ids, :]

        @floop executor for id in current_active_neuron_ids
            axpy!(dz[id], input, dWs[:, id])
        end
    end
end

""" Force sequential on neuron gradient update """
_get_executors(::SlideLayer, default_executor) = default_executor, SequentialEx()
""" Force sequential batch gradient updates """
_get_executors(::AbstractLayer, default_executor) = SequentialEx(), default_executor

function calculate_wgrads!(
    layer::L,
    inputs::AbstractMatrix{Float};
    executor = ThreadedEx(),
) where {L<:AbstractLayer}
    batch_size = size(inputs, 2)
    batch_executor, wgrad_executor = _get_executors(layer, executor)

    @views @floop batch_executor for i = 1:batch_size
        input, active_ids = inputs[:, i], (:)
        _calculate_wgrads!(layer, input, active_ids, i; executor = wgrad_executor)
    end
end

function calculate_wgrads!(
    layer::L,
    inputs::Tuple{Vector{Ids},Vector{T}};
    executor = ThreadedEx(),
) where {L<:AbstractLayer,Ids,T<:FloatVector}
    batch_active_ids, batch_input = inputs
    batch_size = length(batch_active_ids)

    batch_executor, wgrad_executor = _get_executors(layer, executor)

    @views @floop batch_executor for i = 1:batch_size
        input, active_ids = batch_input[i], batch_active_ids[i]
        _calculate_wgrads!(layer, input, active_ids, i; executor = wgrad_executor)
    end
end
