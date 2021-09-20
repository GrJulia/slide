using FLoops: @floop, ThreadedEx

using Slide: FloatVector
using Slide.Network.Optimizers: AbstractOptimizer, optimizer_step!, AdamAttributes
using Slide.Network.Layers: calculate_wgrads!, calculate_error!, get_output, set_error!



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
                Ref(layer.bias[id]),
                layer.weight_gradients[:, id],
                layer.bias_gradients[id, :],
            )
        end
    end
end

function backward!(
    network::SlideNetwork,
    x::AbstractMatrix{Float},
    y_true::Vector{<:FloatVector},
    saved_softmax::Vector{<:FloatVector};
    executor = ThreadedEx(),
)
    batch_size = size(x)[2]

    zero_grads!(network, batch_size)

    @views @floop executor for i = 1:batch_size
        last_layer = network.layers[end]
        active_neuron_ids, _ = get_output(last_layer, i)

        error = gradient(
            typeof(negative_sparse_logit_cross_entropy),
            y_true[i],
            saved_softmax[i],
            sum(y_true[i]),
        )
        set_error!(last_layer, i, active_neuron_ids, error)

        n_layers = length(network.layers)
        for t = n_layers-1:-1:1
            calculate_error!(network.layers[t], network.layers[t+1], i)
        end
    end

    input = x
    for layer in network.layers
        calculate_wgrads!(layer, input; executor)
        input = get_output(layer)
    end
end
