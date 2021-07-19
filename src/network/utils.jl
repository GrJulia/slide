using Statistics
using Slide.Network

function batch_input(
    x::Matrix{Float32},
    y::Matrix{Int},
    batch_size::Int64,
    drop_last::Bool,
)::Vector{Tuple{Matrix{Float32},Matrix{Int}}}
    batches = map(Iterators.partition(axes(x, 2), batch_size)) do columns
        x[:, columns], y[:, columns]
    end
    if drop_last && size(batches[end])[1] < batch_size
        return batches[1:end-1]
    end
    return batches
end

function one_hot(y::Vector, n_labels::Int64 = maximum(y))
    y_categorical = zeros(Int64, n_labels, length(y))
    for (i, label) in enumerate(y)
        y_categorical[label, i] = 1
    end
    y_categorical
end

function cross_entropy(y_pred, y_true)
    -mean(sum(y_true .* log.(y_pred .+ eps()), dims = 1))
end

function empty_neurons_attributes!(network)
    for layer in network.layers
        for neuron in layer.neurons
            neuron.weight_gradients = zeros(size(neuron.weight_gradients))
            neuron.bias_gradients = zeros(size(neuron.bias_gradients))
            neuron.active_inputs = zeros(size(neuron.active_inputs))
            neuron.activation_inputs = zeros(size(neuron.activation_inputs))
        end
    end
end

function numerical_gradient(network, layer_id, neuron_id, weight_index, x, y, epsilon)
    y = forward(x, network)
    backward!(x, y, network)
    backprop_gradient =
        mean(network.layers[layer_id].neurons[neuron_id].weight_gradients, dims = 2)
    network.layers[layer_id].neurons[neuron_id].weight[weight_index] += epsilon
    loss_1 = cross_entropy(forward(x, network), y)
    network.layers[layer_id].neurons[neuron_id].weight[weight_index] -= 2 * epsilon
    loss_2 = cross_entropy(forward(x, network), y)
    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)
    println("Numerical gradient: $numerical_grad")
    println("Manual gradient: $(backprop_gradient[weight_index])")
end
