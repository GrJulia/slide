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

function one_hot(y::Vector{Vector{Float}}, n_labels::Int64 = Int(maximum(maximum.(y))))
    # One hot encoding for datasets used by Slide
    # which turns out to be a mutlilabels classification problem
    y_categorical = zeros(Float, n_labels, length(y))
    for (i, label_vector) in enumerate(y)
        for label in label_vector
            y_categorical[Int(label), i] = 1
        end
    end
    y_categorical
end

function cross_entropy(y_pred::Array{Float}, y_true::Array{Float})
    -mean(sum(y_true .* log.(y_pred .+ eps()), dims = 1))
end

function empty_neurons_attributes!(network::SlideNetwork)
    for layer in network.layers
        for neuron in layer.neurons
            neuron.neuron.weight_gradients = fill!(
                neuron.neuron.weight_gradients,
                zero(eltype(neuron.neuron.weight_gradients)),
            )
            neuron.neuron.bias_gradients = fill!(
                neuron.neuron.bias_gradients,
                zero(eltype(neuron.neuron.bias_gradients)),
            )
            neuron.neuron.active_inputs = fill!(
                neuron.neuron.active_inputs,
                zero(eltype(neuron.neuron.active_inputs)),
            )
            neuron.neuron.activation_inputs = fill!(
                neuron.neuron.activation_inputs,
                zero(eltype(neuron.neuron.activation_inputs)),
            )
        end
    end
end

function numerical_gradient_weights(
    network::SlideNetwork,
    layer_id::Int,
    neuron_id::Int,
    weight_index::Int,
    x_check,
    y_check,
    epsilon::Float,
)
    empty_neurons_attributes!(network)
    y_check_pred, activated_neurons = forward!(x_check, network, false)
    backward!(x_check, y_check_pred, network)
    backprop_gradient =
        sum(network.layers[layer_id].neurons[neuron_id].neuron.weight_gradients, dims = 2)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.weight[weight_index] += epsilon
    y_check_pred_1, activated_neurons_1 = forward!(x_check, network, false)
    loss_1 = negative_sparse_logit_cross_entropy(y_check_pred_1, y_check, activated_neurons_1)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.weight[weight_index] -= 2 * epsilon
    y_check_pred_2, activated_neurons_2 = forward!(x_check, network, false)
    loss_2 = negative_sparse_logit_cross_entropy(y_check_pred_2, y_check, activated_neurons_2)
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
    x_check,
    y_check,
    epsilon::Float,
)
    empty_neurons_attributes!(network)
    y_check_pred, activated_neurons  = forward!(x_check, network, false)
    backward!(x_check, y_check_pred, network)
    backprop_gradient =
        sum(network.layers[layer_id].neurons[neuron_id].neuron.bias_gradients)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.bias += epsilon
    y_check_pred_1, activated_neurons_1 = forward!(x_check, network, false)
    loss_1 = negative_sparse_logit_cross_entropy(y_check_pred_1, y_check, activated_neurons_1)
    empty_neurons_attributes!(network)
    network.layers[layer_id].neurons[neuron_id].neuron.bias -= 2 * epsilon
    y_check_pred_2, activated_neurons_2 = forward!(x_check, network, false)
    loss_2 = negative_sparse_logit_cross_entropy(y_check_pred_2, y_check, activated_neurons_2)
    empty_neurons_attributes!(network)
    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)
    println("Numerical gradient: $numerical_grad")
    println("Manual gradient: $backprop_gradient")
    println("Absolute grad diff: $(abs(numerical_grad - backprop_gradient))")
end
