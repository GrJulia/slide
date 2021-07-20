using LinearAlgebra

function build_activated_neurons_single_sample(
    x::Vector{Float},
    network::SlideNetwork,
    random,
)::Vector{Vector{Int}}
    activated_neuron_ids = []
    for layer in network.layers
        current_hash_table = layer.hash_table
        if random
            input_hash = get_random_hash(current_hash_table, x)
        else
            input_hash = get_deterministic_hash(current_hash_table, x)
        end
        neuron_ids = retrieve_ids_from_bucket(current_hash_table, input_hash)
        push!(activated_neuron_ids, neuron_ids)
    end
    return activated_neuron_ids
end

function forward_single_sample(
    x::Vector{Float},
    network::SlideNetwork,
    activated_neuron_ids::Vector,
    x_index::Int,
)::Vector{Float}
    n_layers = length(network.layers)
    current_input = x
    for i = 1:n_layers
        layer = network.layers[i]
        current_n_neurons = length(layer.neurons)
        layer_activation = layer.layer_activation
        layer_output = zeros(current_n_neurons)
        for neuron_id in activated_neuron_ids[i]
            current_neuron = layer.neurons[neuron_id].neuron
            layer_output[neuron_id] =
                dot(current_input, current_neuron.weight) + current_neuron.bias
        end
        current_input = layer_activation(layer_output)
        for (k, opt_neuron) in enumerate(layer.neurons)
            opt_neuron.neuron.activation_inputs[x_index] = current_input[k]
        end
    end
    return current_input
end

function handle_batch(
    x::Vector{Float},
    network::SlideNetwork,
    i::Int,
    random,
)::Vector{Float}
    activated_neuron_ids = build_activated_neurons_single_sample(x, network, random)
    for j = 1:length(activated_neuron_ids)
        for neuron_id in activated_neuron_ids[j]
            network.layers[j].neurons[neuron_id].neuron.active_inputs[i] = 1
        end
    end
    return forward_single_sample(x, network, activated_neuron_ids, i)
end

function forward!(x::Matrix{Float}, network::SlideNetwork, random = true)::Matrix{Float}
    n_samples = size(x)[2]
    output = zeros(length(network.layers[end].neurons), n_samples)
    Threads.@threads for i = 1:n_samples
        output[:, i] = handle_batch(x[:, i], network, i, random)
    end
    output
end
