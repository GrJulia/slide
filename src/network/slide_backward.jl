using Statistics: mean, Threads

function get_active_neurons(layer::Layer, sample_index::Int)::Vector{Neuron}
    return [neuron for neuron in layer.neurons if neuron.active_inputs[sample_index] == 1]
end

function handle_batch_backward(
    x::SubArray{Float},
    y::SubArray{Float},
    network::SlideNetwork,
    i::Int,
    saved_softmax::Vector{Float},
)
    for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = get_active_neurons(layer, i)
        if l == 1
            previous_activation = x
        else
            previous_activation = [
                previous_neuron.activation_inputs[i] for
                previous_neuron in network.layers[l-1].neurons
            ]
        end
        for (k, neuron) in enumerate(active_neurons)
            if l == length(network.layers)
                dz = gradient(
                    typeof(negative_sparse_logit_cross_entropy),
                    saved_softmax[k],
                    y[neuron.id],
                    sum(y),
                ) # recall that saved_softmax's length is size(active_neurons)
            # sum(y): to handle multiple labels
            else
                da = sum(
                    neuron.bias_gradients[i] * neuron.weight[k] for
                    neuron in network.layers[l+1].neurons
                )
                dz =
                    da *
                    gradient(typeof(layer.layer_activation), neuron.activation_inputs[i])
            end
            neuron.bias_gradients[i] = dz
            neuron.weight_gradients[:, i] = dz .* previous_activation
        end
    end
end

function update_weight!(network::SlideNetwork, optimizer::Optimizer)
    for layer in network.layers
        for neuron in layer.neurons
            optimizer_step!(optimizer, neuron)
        end
    end
end

function backward!(
    x::Matrix{Float},
    y_pred::Matrix{Float},
    network::SlideNetwork,
    saved_softmax::Vector{Vector{Float}},
)
    n_samples = size(x)[2]
    Threads.@threads for i = 1:n_samples
        handle_batch_backward(
            (@view x[:, i]),
            (@view y_pred[:, i]),
            network,
            i,
            saved_softmax[i],
        )
    end
end
