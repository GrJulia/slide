using Zygote
using LinearAlgebra
using Flux
using Flux.Losses: logitcrossentropy
using FLoops: @floop, ThreadedEx

using Slide.Network: SlideNetwork
using Slide: FloatVector

const LayerTrainableParams = Vector{Tuple{Matrix{Float},Vector{Float}}}

function slide_loss(y_true::T, output::U)::Float where {T<:FloatVector,U<:FloatVector}
    logitcrossentropy(y_true, softmax(output))
end

function full_forward(x::T, parameters::LayerTrainableParams) where {T<:FloatVector}
    current_input = x
    for (W, b) in parameters
        current_input = W' * current_input + b
    end
    return current_input
end

function handle_batch_backward_zygote!(
    x::T,
    y_true::P,
    network::SlideNetwork,
    i::Int,
) where {T<:FloatVector,P<:FloatVector}

    parameters = LayerTrainableParams()
    prev_active_neuron_ids = (:)
    @views for layer in network.layers
        active_neuron_ids = layer.active_neuron_ids[i]
        push!(
            parameters,
            (
                layer.weights[prev_active_neuron_ids, active_neuron_ids],
                layer.biases[active_neuron_ids],
            ),
        )

        prev_active_neuron_ids = layer.active_neuron_ids[i]
    end
    slide_gradients =
        Zygote.gradient(p -> slide_loss(y_true, full_forward(x, p)), parameters)[1]
    @views for (k, layer) in enumerate(network.layers)
        w_grad, b_grad = slide_gradients[k]
        if k == 1
            layer.weight_gradients[:, layer.active_neuron_ids[i]] += w_grad
            layer.bias_gradients[layer.active_neuron_ids[i]] += b_grad
        else
            layer.weight_gradients[
                network.layers[k-1].active_neuron_ids[i],
                layer.active_neuron_ids[i],
            ] += w_grad
            layer.bias_gradients[layer.active_neuron_ids[i]] += b_grad
        end

    end
end


function backward_zygote!(
    x::Matrix{Float},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
    executor = ThreadedEx(),
)
    batch_size = size(x)[2]
    @views @floop executor for i = 1:batch_size
        handle_batch_backward_zygote!(x[:, i], y_true[i], network, i)
    end
end
