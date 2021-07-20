function build_network(
    n_layers::Int,
    n_neurons_per_layer::Vector{Int},
    layer_activations::Vector{String},
    input_dim::Int,
    hash_tables::Vector,
    batch_size::Int,
)::SlideNetwork
    network_layers = Vector{Layer}()
    for i = 1:n_layers
        neurons = Vector{Neuron}()
        if i == 1
            current_input_dim = input_dim
        else
            current_input_dim = n_neurons_per_layer[i-1]
        end
        for j = 1:n_neurons_per_layer[i]
            push!(
                neurons,
                Neuron(
                    j,
                    rand(current_input_dim),
                    rand(),
                    zeros(batch_size),
                    zeros(batch_size),
                    zeros(current_input_dim, batch_size),
                    zeros(batch_size),
                    zeros(current_input_dim),
                    0,
                    zeros(current_input_dim),
                    0,
                ),
            )
        end
        layer = Layer(
            i,
            neurons,
            hash_tables[i],
            activation_name_to_function[layer_activations[i]],
        )
        store_neurons_in_bucket(layer.hash_table, layer.neurons)
        push!(network_layers, layer)
    end
    network = SlideNetwork(network_layers)
    return network
end


function build_and_train(
    x::Matrix{Float},
    y::Vector{Int},
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
    batches = batch_input(x, y_cat, batch_size, drop_last)
    output = nothing
    for i = 1:n_iters
        loss = 0
        output = Array{typeof(x[1])}(undef, length(network.layers[end].neurons), 0)
        for (x_batch, y_batch) in batches
            y_batch_pred = forward!(x_batch, network)
            output = hcat(output, y_batch_pred)
            loss += cross_entropy(y_batch_pred, y_batch)
            backward!(x_batch, y_batch_pred, network)
            update_weight!(network, optimizer)
            empty_neurons_attributes!(network)
        end
        println("Iteration $i, Loss $(loss / length(batches))")
    end
    return output, network
end
