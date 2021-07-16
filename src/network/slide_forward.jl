using Base: Int64
using LinearAlgebra


function build_network(
    n_layers::Int64,
    n_neurons_per_layer::Vector{Int64},
    layer_activations::Vector{String},
    input_dim::Int64,
    hash_tables::Vector,
    batch_size::Int64,
)::SlideNetwork
    network_layers = Vector{Layer}()
    for i in 1:n_layers
        neurons = Vector{Neuron}()
        if i == 1
            current_input_dim = input_dim
        else
            current_input_dim = n_neurons_per_layer[i - 1]
        end
        for j = 1:n_neurons_per_layer[i]
            push!(neurons, Neuron(j, rand(current_input_dim), rand(), zeros(batch_size)))
        end
        layer = Layer(
            i,
            neurons,
            hash_tables[i],
            activation_name_to_function[layer_activations[i]],
        )
        store_neurons_in_bucket(layer.hash_table, layer.neurons)
        push!(network_layers , layer)
    end
    network = SlideNetwork(network_layers)
    return network
end

function build_activated_neurons_single_sample(
    x::Vector{Float32},
    network::SlideNetwork,
)::Vector{Vector{Int64}}
activated_neuron_ids = []
    for layer in network.layers
        current_hash_table = layer.hash_table
        input_hash = get_hash(current_hash_table, x)
        neuron_ids = retrieve_ids_from_bucket(current_hash_table, input_hash)
        push!(activated_neuron_ids, neuron_ids)
    end
    return activated_neuron_ids
end

function forward_single_sample(
    x::Vector{Float32},
    network::SlideNetwork,
    activated_neuron_ids::Vector,
)::Vector{Float32}
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
    end
    return current_input
end

function handle_batch(x, network, i)
    activated_neuron_ids = build_activated_neurons_single_sample(x, network)
    for j in 1:length(activated_neuron_ids)
        for neuron_id in activated_neuron_ids[j]
            network.layers[j].neurons[neuron_id].activation_input[i] = 1
        end
    end
    return forward_single_sample(x, network, activated_neuron_ids)
end

function forward(x::Matrix{Float32}, network::SlideNetwork)::Matrix{Float32}
    n_samples = size(x)[2]
    output = zeros(length(network.layers[end].neurons), n_samples)
    Threads.@threads for i = 1:n_samples
        output[:, i] = handle_batch(x[:, i], network, i)
    end
    output
end

function batch_input(
    x::Matrix{Float32},
    batch_size::Int64,
    drop_last::Bool,
)::Vector{Matrix{Float32}}
    batches = map(Iterators.partition(axes(x, 2), batch_size)) do columns
        x[:, columns]
    end
    if drop_last && size(batches[end])[1] < batch_size
        return batches[1:end-1]
    end
    return batches
end
