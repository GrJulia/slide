using Statistics
using Slide.Network

const Batch = Tuple{Matrix{Float},Matrix{Float}}

function batch_input(
    x::Matrix{Float},
    y::Matrix{Float},
    batch_size::Int64,
    drop_last::Bool,
)::Vector{Batch}
    batches = map(Iterators.partition(axes(x, 2), batch_size)) do columns
        (@view x[:, columns]), (@view y[:, columns])
    end
    if drop_last && size(batches[end])[1] < batch_size
        return batches[1:end-1]
    end
    return batches
end

function one_hot(y::Vector, n_labels::Int64 = Int(maximum(y)))
    y_categorical = zeros(Float, n_labels, length(y))
    for (i, label) in enumerate(y)
        y_categorical[Int(label), i] = 1
    end
    y_categorical
end

function cross_entropy(y_pred::Array{Float}, y_true::Array{Float})
    -mean(sum(y_true .* log.(y_pred .+ eps()), dims = 1))
end

function empty_neurons_attributes!(network::SlideNetwork)
    for layer in network.layers
        for neuron in layer.neurons
            neuron.neuron.weight_gradients = zeros(size(neuron.neuron.weight_gradients))
            neuron.neuron.bias_gradients = zeros(size(neuron.neuron.bias_gradients))
            neuron.neuron.active_inputs = zeros(size(neuron.neuron.active_inputs))
            neuron.neuron.activation_inputs = zeros(size(neuron.neuron.activation_inputs))
        end
    end
end

function numerical_gradient_weights(
    network::SlideNetwork,
    layer_id::Int,
    neuron_id::Int,
    weight_index::Int,
    x_check::Vector{Float},
    y_check::Vector{Float},
    epsilon::Float,
)
    empty_neurons_attributes!(network)
    y_check_pred = handle_batch((@view x_check[:]), network, 1, false)
    handle_batch_backward((@view x_check[:]), (@view y_check_pred[:]), network, 1)
    backprop_gradient =
        sum(network.layers[layer_id].neurons[neuron_id].neuron.weight_gradients, dims = 2)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.weight[weight_index] += epsilon
    loss_1 = cross_entropy(handle_batch((@view x_check[:]), network, 1, false), y_check)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.weight[weight_index] -= 2 * epsilon
    loss_2 = cross_entropy(handle_batch((@view x_check[:]), network, 1, false), y_check)
    empty_neurons_attributes!(network)
    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)
    println("Numerical gradient: $numerical_grad")
    println("Manual gradient: $(backprop_gradient[weight_index])")
    println("Absolute grad diff: $(abs(numerical_grad - backprop_gradient[weight_index]))")
end

function numerical_gradient_bias(
    network::SlideNetwork,
    layer_id::Int,
    neuron_id::Int,
    x_check::Vector{Float},
    y_check::Vector{Float},
    epsilon::Float,
)
    empty_neurons_attributes!(network)
    y_check_pred = handle_batch((@view x_check[:]), network, 1, false)
    handle_batch_backward((@view x_check[:]), (@view y_check_pred[:]), network, 1)
    backprop_gradient =
        sum(network.layers[layer_id].neurons[neuron_id].neuron.bias_gradients)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.bias += epsilon
    loss_1 = cross_entropy(handle_batch((@view x_check[:]), network, 1, false), y_check)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.bias -= 2 * epsilon
    loss_2 = cross_entropy(handle_batch((@view x_check[:]), network, 1, false), y_check)
    empty_neurons_attributes!(network)
    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)
    println("Numerical gradient: $numerical_grad")
    println("Manual gradient: $backprop_gradient")
    println("Absolute grad diff: $(abs(numerical_grad - backprop_gradient))")
end
