using Zygote
using LinearAlgebra
using Flux
using Flux.Losses: logitcrossentropy
using FLoops: @floop, ThreadedEx

const FloatVector = AbstractVector{Float}

function slide_loss(y_true, output)
    logitcrossentropy(y_true, softmax(output))
end

function full_forward(x, parameters)
    current_input = x
    for p in parameters
        current_input = p[1]' * current_input + p[2]
    end
    return current_input
end

function handle_batch_backward_zygote!(
    x::T,
    y::U,
    y_true::P,
    network::SlideNetwork,
    i::Int,
) where {T<:FloatVector,P<:FloatVector,U<:FloatVector}

    parameters = []
    for (k, layer) in enumerate(network.layers)
        if k == 1
            push!(parameters, (layer.weights[:, layer.active_neuron_ids[i]], layer.biases[layer.active_neuron_ids[i]]))
        else
            push!(parameters, (layer.weights[network.layers[k - 1].active_neuron_ids[i], layer.active_neuron_ids[i]], layer.biases[layer.active_neuron_ids[i]]))
        end
    end

    slide_gradients = Zygote.gradient(
        (p) -> slide_loss(y_true, full_forward(x, p)),
        parameters
    )[1]

    for (k, layer) in enumerate(network.layers)
        if k == 1
            layer.weight_gradients[:, layer.active_neuron_ids[i]] += slide_gradients[k][1]
            layer.bias_gradients[layer.active_neuron_ids[i]] += slide_gradients[k][2]
        else
            layer.weight_gradients[network.layers[k - 1].active_neuron_ids[i], layer.active_neuron_ids[i]] += slide_gradients[k][1]
            layer.bias_gradients[layer.active_neuron_ids[i]] += slide_gradients[k][2]
        end
        
    end    
end


function backward_zygote!(
    x::Matrix{Float},
    y_pred::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
    executor = ThreadedEx(),
)
    n_samples = size(x)[2]
    @views @floop executor for i = 1:n_samples
        handle_batch_backward_zygote!(x[:, i], y_pred[i], y_true[i], network, i)
    end
end
