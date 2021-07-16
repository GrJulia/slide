function handle_batch_backward(x, y, network, loss, i)
    for l in length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = [neuron for neuron in layer.neurons if neuron.active_inputs[i] == 1]
        for (k, neuron) in enumerate(active_neurons)
            if l == length(network.layers)
                #neuron.gradients[i] = neurons.activation_inputs[i] - y[k]
                bias_gradient = neuron.activation_inputs[i] - y[k]
                weight_gradient = bias_gradient .* [previous_neuron.activation_inputs[i] for previous_neuron in network.layers[l - 1].neurons]
                neuron.bias_gradients[i] = bias_gradient
                neuron.weight_gradients[:, i] = weight_gradient 
            else
                if l == 1
                    previous_activation = x
                else
                    previous_activation = [previous_neuron.activation_inputs[i] for previous_neuron in network.layers[l - 1].neurons]
                end
                bias_gradient = neuron.activation_inputs[i] * (neuron.activation_inputs[i] - 1) * network.layers[l + 1].neurons[1].bias_gradients[i] ## FIX ME
                weight_gradient = bias_gradient .* previous_activation
                neuron.bias_gradients[i] = bias_gradient
                neuron.weight_gradients[:, i] = weight_gradient 
            end
        end
    end
end

function backward!(x, y_pred, y_true, loss, network)
    n_samples = size(x)[2]
    Threads.@threads for i = 1:n_samples
        handle_batch_backward(x[:, i], y_pred[:, i], network, loss, i)
    end
end