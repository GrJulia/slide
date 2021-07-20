using Statistics: mean, Threads

function handle_batch_backward(
    x::Vector{Float},
    y::Vector{Float},
    network::SlideNetwork,
    i::Int,
)
    for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = [
            opt_neuron.neuron for
            opt_neuron in layer.neurons if opt_neuron.neuron.active_inputs[i] == 1
        ]
        if l == 1
            previous_activation = x
        else
            previous_activation = [
                previous_neuron.neuron.activation_inputs[i] for
                previous_neuron in network.layers[l-1].neurons
            ]
        end
        for (k, neuron) in enumerate(active_neurons) # refactor
            if l == length(network.layers)
                dz = neuron.activation_inputs[i] - y[k]
                #dz = gradient(typeof(layer.layer_activation), neuron.activation_inputs[i],  y[k])
            else
                da = sum(
                    neuron.neuron.bias_gradients[i] * neuron.neuron.weight[k] for
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
            optimizer_step(optimizer, neuron)
        end
    end
end

function backward!(x::Matrix{Float}, y_pred::Matrix{Float}, network::SlideNetwork)
    n_samples = size(x)[2]
    Threads.@threads for i = 1:n_samples
        handle_batch_backward(x[:, i], y_pred[:, i], network, i)
    end
end
