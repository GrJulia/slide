using Statistics: mean, Threads

function handle_batch_backward(
    x::Vector{Float},
    y::Vector{Float},
    network::SlideNetwork,
    i::Int,
)
    for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons =
            [neuron for neuron in layer.neurons if neuron.active_inputs[i] == 1]
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
                dz = neuron.activation_inputs[i] - y[k]
            else
                da = sum(
                    neuron.bias_gradients[i] * neuron.weight[k] for
                    neuron in network.layers[l+1].neurons
                )
                dz = da * neuron.activation_inputs[i] * (1 - neuron.activation_inputs[i])
            end
            neuron.bias_gradients[i] = dz
            neuron.weight_gradients[:, i] = dz .* previous_activation
        end
    end
end

function update_weight!(network::SlideNetwork, learning_rate::Float)
    for layer in network.layers
        for neuron in layer.neurons
            neuron.weight .-= learning_rate * mean(neuron.weight_gradients, dims = 2)[:, 1]
            neuron.bias -= learning_rate * mean(neuron.bias_gradients)
        end
    end
end

function backward!(x::Matrix{Float}, y_pred::Matrix{Float}, network::SlideNetwork)
    n_samples = size(x)[2]
    Threads.@threads for i = 1:n_samples
        handle_batch_backward(x[:, i], y_pred[:, i], network, i)
    end
end
