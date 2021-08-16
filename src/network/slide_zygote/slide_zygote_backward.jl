using Zygote
using LinearAlgebra
using Flux
using Flux.Losses: logitcrossentropy

function handle_batch_backward_zygote!(
    x,
    y,
    y_true,
    network::SlideNetwork,
    i::Int,
)
    da = Zygote.gradient(output -> logitcrossentropy(y_true, softmax(output)), y)[1]
    @inbounds for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = layer.active_neuron_ids[i]

        if l == 1
            previous_neurons = Vector{Id}(1:length(x))
        else
            previous_neurons = network.layers[l-1].active_neuron_ids[i]
        end

        if l < length(network.layers)
            da = [sum([da[j] * network.layers[l + 1].neurons[j].weight[k] for j = 1:length(da)]) for k in active_neurons]
        else

        end

        for (k, neuron) in enumerate(view(layer.neurons, active_neurons))
            neuron.is_active = true
            neuron.weight_gradients[previous_neurons] += neuron.grad_output_w[i] * da[k]
            neuron.bias_gradients[i] += neuron.grad_output_b[i] * da[k]
        end
    end
end


function backward_zygote!(
    x::Matrix{Float},
    y_pred::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
)
    n_samples = size(x)[2]
    @views for i = 1:n_samples
        handle_batch_backward_zygote!(x[:, i], y_pred[i], y_true[i], network, i)
    end
end
