using LinearAlgebra
using Slide.LSH: retrieve


function forward_single_sample(
    x::SubArray{Float},
    network::SlideNetwork,
    x_index::Int,
)::Vector{Float}
    n_layers = length(network.layers)
    current_input = x
    for i = 1:n_layers
        # compute activated neurons with current_input
        layer = network.layers[i]

        #get activated neurons and mark them as changed
        activated_neuron_ids =
            [x for x in retrieve(layer.hash_tables.lsh, @view current_input[:])]
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

function forward!(x::Array{Float}, network::SlideNetwork)
    n_samples = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    output = zeros(length(network.layers[end].neurons), n_samples)

    Threads.@threads for i = 1:n_samples
        output[:, i] = forward_single_sample((@view x[:, i]), network, i)
    end

    output
end

function predict_class(x::Array{Float}, network::SlideNetwork)
    y_pred, _ = forward!(x, network)
    return mapslices(argmax, y_pred, dims = 1)
end
