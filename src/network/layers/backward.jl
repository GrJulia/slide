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
        grad = get_bias_gradients(next_layer, x_index)[active_neuron_ids]
        weights = get_weights(next_layer)[current_active_neuron_ids, active_neuron_ids]

        get_bias_gradients(layer, x_index)[current_active_neuron_ids] .=
            (weights * grad) .* gradient(F, output)
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

    @floop executor for id in current_active_neuron_ids
        @views axpy!(
            get_bias_gradients(layer, x_index)[id],
            input,
            get_weight_gradients(layer)[previous_active_ids, id],
        )
    end
end

""" Force sequential on neuron gradient update """
_get_executors(::AbstractLayer, default_executor) = default_executor, SequentialEx()
""" Force sequential batch gradient updates """
_get_executors(::Dense, default_executor) = SequentialEx(), default_executor

function calculate_wgrads!(
    layer::L,
    inputs::Matrix{Float};
    executor = ThreadedEx(),
) where {L<:AbstractLayer}
    batch_size = size(inputs, 2)
    batch_executor, wgrad_executor = _get_executors(layer, executor)

    @views @floop batch_executor for i = 1:batch_size
        in, active_ids = inputs[:, i], (:)
        _calculate_wgrads!(layer, in, active_ids, i; executor = wgrad_executor)
    end
end

function calculate_wgrads!(
    layer::L,
    inputs::Tuple{Vector{Ids},Vector{T}};
    executor = ThreadedEx(),
) where {L<:AbstractLayer,Ids,T<:FloatVector}
    ids, input = inputs
    batch_size = length(ids)

    batch_executor, wgrad_executor = _get_executors(layer, executor)

    @views @floop batch_executor for i = 1:batch_size
        in, active_ids = input[i], ids[i]
        _calculate_wgrads!(layer, in, active_ids, i; executor = wgrad_executor)
    end
end
