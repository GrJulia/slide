using LinearAlgebra
using ..LSH: retrieve


function forward_single_sample(
    x::SubArray{Float},
    y_true::SubArray{Float},
    network::SlideNetwork,
    x_index::Int,
    use_all_true_labels::Bool,
)::Vector{Float}
    n_layers = length(network.layers)
    current_input = x
    for i = 1:n_layers
        # compute activated neurons with current_input
        layer = network.layers[i]

        activated_neuron_ids =
            [x for x in retrieve(layer.hash_tables.lsh, @view current_input[:])]
        if use_all_true_labels && (i == length(network.layers))
            activated_neuron_ids =
                sort!(unique(vcat(activated_neuron_ids, findall(>(0), y_true))))
        end
        mark_ids!(layer.hash_tables, activated_neuron_ids)

        for neuron_id in activated_neuron_ids
            layer.neurons[neuron_id].active_inputs[x_index] = 1
        end

        current_n_neurons = length(layer.neurons)
        layer_activation = layer.layer_activation
        layer_output = zeros(current_n_neurons)
        for neuron_id in activated_neuron_ids
            current_neuron = layer.neurons[neuron_id]
            layer_output[neuron_id] =
                dot(current_input, current_neuron.weight) + current_neuron.bias
        end
        current_input = layer_activation(layer_output)
        for (k, neuron) in enumerate(layer.neurons)
            neuron.pre_activation_inputs[x_index] = layer_output[k]
            neuron.activation_inputs[x_index] = current_input[k]
        end
    end
    return current_input
end

function forward!(
    x::Array{Float},
    y_true::Array{Float},
    network::SlideNetwork,
    use_all_true_labels::Bool = true,
)
    n_samples = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    output = zeros(length(network.layers[end].neurons), n_samples)

    Threads.@threads for i = 1:n_samples
        output[:, i] = forward_single_sample(
            (@view x[:, i]),
            (@view y_true[:, i]),
            network,
            i,
            use_all_true_labels,
        )
    end

    output
end

function predict_class(x::Array{Float}, y_true::Array{Float}, network::SlideNetwork)
    y_pred, _ = forward!(x, y_true, network)
    return mapslices(argmax, y_pred, dims = 1)
end
