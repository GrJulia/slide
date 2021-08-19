using FLoops: @floop, ThreadedEx
using Base.Threads: nthreads, threadid

using Slide: FloatVector
using Slide.Network.Optimizers: AbstractOptimizer, optimizer_step!, AdamAttributes
using Slide.Network.Layers: backward_single_sample!


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
                layer.bias_gradients[:, id],
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

    zero_grads!(network, batch_size)

    max_size = maximum(map(l -> length(l.biases), network.layers))
    loss_storage = Array{Float}(undef, max_size, nthreads())

    @views @floop executor for i = 1:batch_size
        loss_storage[begin:length(y_true[i]), threadid()] = gradient(
            typeof(negative_sparse_logit_cross_entropy),
            y_true[i],
            saved_softmax[i],
            sum(y_true[i]),
        )
        for layer in network.layers[2:end]
            prev_layer = network.layers[layer.id-1]
            backward_single_sample!(
                layer,
                loss_storage[:, threadid()],
                (prev_layer.output[i], prev_layer.active_neuron_ids[i]),
                loss_storage[:, threadid()],
                i,
            )
        end

        first_layer = network.layers[begin]
        backward_single_sample!(first_layer, loss_storage[:, threadid()], x[:, i], i)
    end
end
