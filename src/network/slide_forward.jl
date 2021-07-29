using LinearAlgebra

function build_activated_neurons_single_sample(
    x::SubArray{Float},
    layer::Layer,
    random::Bool,
)::Vector{Int}
    current_hash_table = layer.hash_table
    input_hash =
        random ? get_random_hash(current_hash_table, x) :
        get_deterministic_hash(current_hash_table, x)
    return retrieve_ids_from_bucket(current_hash_table, input_hash)
end

function forward_single_sample(
    x::SubArray{Float},
    network::SlideNetwork,
    x_index::Int,
    random::Bool,
)::Tuple{Vector{Float},Vector{Id}}
    n_layers = length(network.layers)
    current_input = x
    activated_neuron_ids = nothing
    for i = 1:n_layers
        # compute activated neurons with current_input
        layer = network.layers[i]

        activated_neuron_ids =
            build_activated_neurons_single_sample((@view current_input[:]), layer, random)
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
    return current_input, activated_neuron_ids
end

function forward!(x::Array{Float}, network::SlideNetwork, random::Bool = false)
    n_samples = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    output = zeros(length(network.layers[end].neurons), n_samples)
    last_layer_activated_neuron_ids = Vector{Vector{Id}}(undef, n_samples)
    Threads.@threads for i = 1:n_samples
        output[:, i], last_layer_activated_neuron_ids_batch =
            forward_single_sample((@view x[:, i]), network, i, random)
        last_layer_activated_neuron_ids[i] = last_layer_activated_neuron_ids_batch
    end
    output, last_layer_activated_neuron_ids
end

function predict_class(x::Array{Float}, network::SlideNetwork)
    y_pred, _ = forward!(x, network, false)
    return mapslices(argmax, y_pred, dims = 1)
end
