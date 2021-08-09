using Statistics: mean, Threads


function handle_batch_backward(
    x::SubArray{Float},
    y::SubArray{Float},
    y_true::SubArray{Float},
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
        active_neuron_ids = [neuron.id for neuron in active_neurons]
        for (k, neuron) in enumerate(active_neurons)
            if l == length(network.layers)
                dz = gradient(
                    typeof(negative_sparse_logit_cross_entropy),
                    y_true[neuron.id],
                    saved_softmax[k],
                    sum(y_true[active_neuron_ids]),
                ) # recall that saved_softmax's length is size(active_neurons)
            # sum(y_true[active_neuron_ids]): to handle multiple labels
            else
                da = sum(
                    next_neuron.bias_gradients[i] * next_neuron.weight[neuron.id] for
                    next_neuron in network.layers[l+1].neurons
                ) # we could only sum over the active neurons in layer l+1, but 
                # here, if a neuron is not active, we're just summing 0
                dz =
                    da * gradient(
                        typeof(layer.layer_activation),
                        neuron.pre_activation_inputs[i],
                    )
            end
            neuron.bias_gradients[i] = dz
            neuron.weight_gradients .+=
                dz .* previous_activation ./ length(neuron.bias_gradients)
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
    y_true::Matrix{Float},
    network::SlideNetwork,
    saved_softmax::Vector{Vector{Float}},
)
    n_samples = size(x)[2]
    Threads.@threads for i = 1:n_samples
        handle_batch_backward(
            (@view x[:, i]),
            (@view y_pred[:, i]),
            (@view y_true[:, i]),
            network,
            i,
            saved_softmax[i],
        )
    end
end
