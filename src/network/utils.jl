using Statistics
using FLoops: SequentialEx

using Slide.Network
using Slide: Float, Id

const Batch = Tuple{Matrix{Float},Matrix{Float}}

function batch_input(x, y, batch_size::Int, drop_last::Bool)::Vector{Batch}
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

function select_by_ids(output, ids)
    [view(output, ids[i], i) for i = 1:size(output, 2)]
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
    activated_neurons, y_check_pred =
        forward!(network, x_check; y_true = y_check, executor = SequentialEx())
    y_check_active = select_by_ids(y_check, activated_neurons)
    _, probs = negative_sparse_logit_cross_entropy(y_check_pred, y_check_active)

    backward!(network, x_check, y_check_active, probs; executor = SequentialEx())
    backprop_gradient = copy(network.layers[layer_id].weight_gradients[:, neuron_id])

    # Computing numerical weight gradient
    network.layers[layer_id].weights[weight_index, neuron_id] += epsilon
    activated_neurons_1, y_check_pred_1 =
        forward!(network, x_check; y_true = y_check, executor = SequentialEx())
    y_check_active_1 = select_by_ids(y_check, activated_neurons_1)
    loss_1, _ = negative_sparse_logit_cross_entropy(y_check_pred_1, y_check_active_1)

    network.layers[layer_id].weights[weight_index, neuron_id] -= 2 * epsilon
    activated_neurons_2, y_check_pred_2 =
        forward!(network, x_check; y_true = y_check, executor = SequentialEx())
    y_check_active_2 = select_by_ids(y_check, activated_neurons_2)
    loss_2, _ = negative_sparse_logit_cross_entropy(y_check_pred_2, y_check_active_2)

    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)

    network.layers[layer_id].weights[weight_index, neuron_id] += epsilon

    abs(numerical_grad - backprop_gradient[weight_index])
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
    activated_neurons, y_check_pred =
        forward!(network, x_check; y_true = y_check, executor = SequentialEx())
    y_check_active = select_by_ids(y_check, activated_neurons)
    _, probs = negative_sparse_logit_cross_entropy(y_check_pred, y_check_active)

    backward!(network, x_check, y_check_active, probs; executor = SequentialEx())
    backprop_gradient = mean(network.layers[layer_id].bias_gradients[neuron_id, :])

    # Computing numerical bias gradient
    network.layers[layer_id].bias[neuron_id] += epsilon
    activated_neurons_1, y_check_pred_1 =
        forward!(network, x_check; y_true = y_check, executor = SequentialEx())
    y_check_active_1 = select_by_ids(y_check, activated_neurons_1)
    loss_1, _ = negative_sparse_logit_cross_entropy(y_check_pred_1, y_check_active_1)

    network.layers[layer_id].bias[neuron_id] -= 2 * epsilon
    y_check_pred_2, activated_neurons_2 =
        forward!(network, x_check; y_true = y_check, executor = SequentialEx())
    y_check_active_2 = select_by_ids(y_check, activated_neurons_2)
    loss_2, _ = negative_sparse_logit_cross_entropy(y_check_pred_2, y_check_active_2)

    numerical_grad = (loss_1 - loss_2) / (2 * epsilon)

    network.layers[layer_id].bias[neuron_id] += epsilon

    abs(numerical_grad - backprop_gradient)
end


function compute_accuracy(
    network::SlideNetwork,
    test_set,
    n_batch_test::Int,
    topk::Int,
)::Float
    accuracy = zero(Float)
    for (x_test, y_test) in first(test_set, n_batch_test)
        class_predictions = predict_class(x_test, y_test, network, topk)
        accuracy += batch_accuracy(y_test, class_predictions, topk)
    end
    return accuracy / n_batch_test
end

function batch_accuracy(y_test, class_predictions::Array{Int}, topk::Int)::Float
    batch_size = size(y_test)[end]
    accuracy = zero(Float)
    for i = 1:batch_size
        accuracy +=
            count(x -> x > 0, y_test[:, i][class_predictions[i]]; init = zero(Float)) / topk
    end
    return accuracy / batch_size
end
