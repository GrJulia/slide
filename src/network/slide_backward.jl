using FLoops: @floop, ThreadedEx
using LinearAlgebra.BLAS: axpy!, dot

using Slide: FloatVector
using Slide.Network.Layers: prep_backprop!
using Slide.Network.Optimizers: AbstractOptimizer, optimizer_step!, AdamAttributes

function handle_batch_backward(
    x::T,
    y::U,
    y_true::P,
    network::SlideNetwork,
    i::Int,
    saved_softmax::Vector{Float},
) where {T<:FloatVector,P<:FloatVector,U<:FloatVector}
    @inbounds for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = layer.active_neuron_ids[i]

        if l == 1
            previous_activation = x
            previous_neurons = Vector{Id}(1:length(x))
        else
            previous_activation = network.layers[l-1].output[i]
            previous_neurons = network.layers[l-1].active_neuron_ids[i]
        end

        for (k, neuron_id) in enumerate(active_neurons)
            layer.is_neuron_active[neuron_id] = true
            if l == length(network.layers)
                # recall that saved_softmax's length is size(active_neurons)
                # sum(y_true): to handle multiple labels
                dz = gradient(
                    typeof(negative_sparse_logit_cross_entropy),
                    y_true[k],
                    saved_softmax[k],
                    sum(y_true),
                )
            else
                # we could only sum over the active neurons in layer l+1, but
                # here, if a neuron is not active, we're just summing 0
                next_layer = network.layers[l+1]
                next_layer_active_neurons_ids = next_layer.active_neuron_ids[i]

                b_gradients =
                    @view next_layer.bias_gradients[next_layer_active_neurons_ids, i]
                n_weights =
                    @view next_layer.weights[neuron_id, next_layer_active_neurons_ids]

                da = sum(b_gradients .* n_weights)
                dz = da * gradient(typeof(layer.layer_activation), layer.output[i][k])
            end

            layer.bias_gradients[neuron_id, i] = dz
            dz = dz / length(layer.bias_gradients[neuron_id, :])
            @views axpy!(
                dz,
                previous_activation,
                layer.weight_gradients[previous_neurons, neuron_id],
            )
        end
    end
end

function update_weight!(
    network::SlideNetwork,
    optimizer::Opt;
    executor = ThreadedEx(),
) where {Opt<:AbstractOptimizer}
    for layer in network.layers
        @floop executor for (id, is_active) in enumerate(layer.is_neuron_active)
            !is_active && continue

            @views optimizer_step!(
                optimizer,
                layer.opt_attr,
                id,
                layer.weights[:, id],
                Ref(layer.biases[id]),
                layer.weight_gradients[:, id],
                layer.bias_gradients[id, :],
            )
        end
    end
end

function backward!(
    x::Matrix{Float},
    y_pred::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
    saved_softmax::Vector{<:FloatVector};
    executor = ThreadedEx(),
)
    batch_size = size(x)[2]

    for layer in network.layers
        prep_backprop!(layer, batch_size)
    end

    @views @floop executor for i = 1:batch_size
        handle_batch_backward(x[:, i], y_pred[i], y_true[i], network, i, saved_softmax[i])
    end
end
