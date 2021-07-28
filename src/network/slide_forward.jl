using LinearAlgebra

function build_activated_neurons_single_sample(
    x::SubArray{Float},
    network::SlideNetwork,
    random::Bool,
)::Vector{Vector{Int}}
    activated_neuron_ids = []
    for layer in network.layers
        current_hash_table = layer.hash_table
        input_hash =
            random ? get_random_hash(current_hash_table, x) :
            get_deterministic_hash(current_hash_table, x)
        neuron_ids = retrieve_ids_from_bucket(current_hash_table, input_hash)
        push!(activated_neuron_ids, neuron_ids)
    end
    return activated_neuron_ids
end

function forward_single_sample(
    x::SubArray{Float},
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

function handle_batch(x::SubArray{Float}, network::SlideNetwork, i::Int, random)
    activated_neuron_ids = build_activated_neurons_single_sample(x, network, random)
    for j = 1:length(activated_neuron_ids)
        for neuron_id in activated_neuron_ids[j]
            network.layers[j].neurons[neuron_id].active_inputs[i] = 1
        end
    end
    return forward_single_sample(x, network, activated_neuron_ids, i),
    activated_neuron_ids[end]
end

function forward!(x::Array{Float}, network::SlideNetwork, random::Bool = false)
    n_samples = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    output = zeros(length(network.layers[end].neurons), n_samples)
    last_layer_activated_neuron_ids = Vector{Vector{Id}}(undef, n_samples)
    Threads.@threads for i = 1:n_samples
        output[:, i], last_layer_activated_neuron_ids_batch =
            handle_batch((@view x[:, i]), network, i, random)
        last_layer_activated_neuron_ids[i] = last_layer_activated_neuron_ids_batch
    end
    output, last_layer_activated_neuron_ids
end

function predict_class(x::Array{Float}, network::SlideNetwork)
    y_pred, _ = forward!(x, network, false)
    return mapslices(argmax, y_pred, dims = 1)
end
