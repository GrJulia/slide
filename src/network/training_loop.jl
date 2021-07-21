function build_network(
    n_layers::Int,
    n_neurons_per_layer::Vector{Int},
    layer_activations::Vector{String},
    input_dim::Int,
    hash_tables::Vector,
    batch_size::Int,
)::SlideNetwork
    network_layers = Vector{Layer}()
    for layer_id = 1:n_layers
        neurons = Vector{OptimizerNeuron{AdamAttributes}}()
        if layer_id == 1
            current_input_dim = input_dim
        else
            current_input_dim = n_neurons_per_layer[layer_id-1]
        end
        for neuron_id = 1:n_neurons_per_layer[layer_id]
            push!(
                neurons,
                OptimizerNeuron(
                    Neuron(neuron_id, batch_size, current_input_dim),
                    AdamAttributes(current_input_dim),
                ),
            )
        end
        layer = Layer(
            layer_id,
            neurons,
            hash_tables[layer_id],
            activation_name_to_function[layer_activations[layer_id]],
        )
        store_neurons_in_bucket(layer.hash_table, layer.neurons)
        push!(network_layers, layer)
    end
    return SlideNetwork(network_layers)
end


function build_and_train(
    x::Matrix{Float},
    y::Vector{Float},
    n_iters::Int,
    batch_size::Int,
    drop_last::Bool,
    network_params::Dict,
    learning_rate::Float,
)
    network = build_network(
        network_params["n_layers"],
        network_params["n_neurons_per_layer"],
        network_params["layer_activations"],
        network_params["input_dim"],
        network_params["hash_tables"],
        batch_size,
    )
    optimizer = AdamOptimizer(eta = learning_rate)
    y_cat = one_hot(y)
    training_batches = batch_input(x, y_cat, batch_size, drop_last)
    output = nothing
    for i = 1:n_iters
        loss = 0
        output = Array{typeof(x[1])}(undef, length(network.layers[end].neurons), 0)
        for (x_batch, y_batch) in training_batches
            y_batch_pred = forward!(x_batch, network)
            output = hcat(output, y_batch_pred)
            loss += cross_entropy(y_batch_pred, y_batch)
            backward!(x_batch, y_batch_pred, network)
            update_weight!(network, optimizer)
            empty_neurons_attributes!(network)
        end
        println("Iteration $i, Loss $(loss / length(training_batches))")
    end
    return output, network
end
