using Statistics
using Slide.Network

const Batch = Tuple{Matrix{Float},Matrix{Float}}

function get_active_neurons(layer::Layer, sample_index::Int)::Vector{Neuron}
    return [neuron for neuron in layer.neurons if neuron.active_inputs[sample_index] == 1]
end

function get_active_neurons_id(network::SlideNetwork, layer_id::Id)::Vector{Vector{Id}}
    batch_size = length(network.layers[1].neurons[1].active_inputs)

    activated = Vector{Vector{Id}}(undef, batch_size)
    for i = 1:batch_size
        activated[i] = map(n -> n.id, get_active_neurons(network.layers[layer_id], i))
    end

    activated
end

function batch_input(
    x::Matrix{Float},
    y::Matrix{Float},
    batch_size::Int,
    drop_last::Bool,
)::Vector{Batch}
    @views batches = map(Iterators.partition(axes(x, 2), batch_size)) do columns
        x[:, columns], y[:, columns]
    end
    if drop_last && size(batches[end])[1] < batch_size
        return batches[1:end-1]
    end
    return batches
end

function one_hot(y::Vector, n_labels::Int = Int(maximum(y)))
    y_categorical = zeros(Float, n_labels, length(y))
    for (i, label) in enumerate(y)
        y_categorical[Int(label), i] = 1
    end
    y_categorical
end

function one_hot(y::Vector{Vector{Float}}, n_labels::Int = Int(maximum(maximum.(y))))
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

function zero_neuron_attributes!(network::SlideNetwork)
    for layer in network.layers
        for neuron in layer.neurons
            neuron.weight_gradients =
                fill!(neuron.weight_gradients, zero(eltype(neuron.weight_gradients)))
            neuron.bias_gradients =
                fill!(neuron.bias_gradients, zero(eltype(neuron.bias_gradients)))
            neuron.active_inputs =
                fill!(neuron.active_inputs, zero(eltype(neuron.active_inputs)))
            neuron.activation_inputs =
                fill!(neuron.activation_inputs, zero(eltype(neuron.activation_inputs)))
            neuron.activation_inputs = fill!(
                neuron.pre_activation_inputs,
                zero(eltype(neuron.pre_activation_inputs)),
            )
        end
    end
end

function numerical_gradient_weights(
    network::SlideNetwork,
    layer_id::Int,
    neuron_id::Int,
    weight_index::Int,
    x_check::Array{Float},
    y_check::Array{Float},
    epsilon::Float,
)
    # Computing weight gradient from backpropagation
    zero_neuron_attributes!(network)
    y_check_pred = forward!(x_check, network)
    activated_neurons = get_active_neurons_id(network, length(network.layers))
    _, probs = negative_sparse_logit_cross_entropy(y_check_pred, y_check, activated_neurons)
    backward!(x_check, y_check_pred, y_check, network, probs)
    backprop_gradient =
        mean(network.layers[layer_id].neurons[neuron_id].weight_gradients, dims = 2)

    zero_neuron_attributes!(network)

    # Computing numerical weight gradient

    network.layers[layer_id].neurons[neuron_id].weight[weight_index] += epsilon
    y_check_pred_1 = forward!(x_check, network)
    activated_neurons_1 = get_active_neurons_id(network, length(network.layers))
    loss_1, _ =
        negative_sparse_logit_cross_entropy(y_check_pred_1, y_check, activated_neurons_1)

    zero_neuron_attributes!(network)

    network.layers[layer_id].neurons[neuron_id].weight[weight_index] -= 2 * epsilon
    y_check_pred_2 = forward!(x_check, network)
    activated_neurons_2 = get_active_neurons_id(network, length(network.layers))
    loss_2, _ =
        negative_sparse_logit_cross_entropy(y_check_pred_2, y_check, activated_neurons_2)


    zero_neuron_attributes!(network)
    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)

    network.layers[layer_id].neurons[neuron_id].weight[weight_index] += epsilon

    return abs(numerical_grad - backprop_gradient[weight_index])
end

function numerical_gradient_bias(
    network::SlideNetwork,
    layer_id::Int,
    neuron_id::Int,
    x_check::Array{Float},
    y_check::Array{Float},
    epsilon::Float,
)
    # Computing bias gradient from backpropagation
    zero_neuron_attributes!(network)
    y_check_pred = forward!(x_check, network)
    activated_neurons = get_active_neurons_id(network, length(network.layers))
    _, probs = negative_sparse_logit_cross_entropy(y_check_pred, y_check, activated_neurons)
    backward!(x_check, y_check_pred, y_check, network, probs)
    backprop_gradient = mean(network.layers[layer_id].neurons[neuron_id].bias_gradients)
    zero_neuron_attributes!(network)

    # Computing numerical bias gradient

    network.layers[layer_id].neurons[neuron_id].bias += epsilon
    y_check_pred_1 = forward!(x_check, network)
    activated_neurons_1 = get_active_neurons_id(network, length(network.layers))
    loss_1, _ =
        negative_sparse_logit_cross_entropy(y_check_pred_1, y_check, activated_neurons_1)

    zero_neuron_attributes!(network)

    network.layers[layer_id].neurons[neuron_id].bias -= 2 * epsilon
    y_check_pred_2 = forward!(x_check, network)
    activated_neurons_2 = get_active_neurons_id(network, length(network.layers))
    loss_2, _ =
        negative_sparse_logit_cross_entropy(y_check_pred_2, y_check, activated_neurons_2)
    zero_neuron_attributes!(network)
    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)

    network.layers[layer_id].neurons[neuron_id].bias += epsilon
    return abs(numerical_grad - backprop_gradient)
end
