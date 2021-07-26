using Slide.Network: Batch, Float

function build_network(network_params::Dict, batch_size::Int)::SlideNetwork
    network_layers = Vector{Layer}()
    for layer_id = 1:network_params["n_layers"]
        neurons = Vector{OptimizerNeuron{AdamAttributes}}()
        if layer_id == 1
            current_input_dim = network_params["input_dim"]
        else
            current_input_dim = network_params["n_neurons_per_layer"][layer_id-1]
        end
        for neuron_id = 1:network_params["n_neurons_per_layer"][layer_id]
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
            network_params["hash_tables"][layer_id],
            activation_name_to_function[network_params["layer_activations"][layer_id]],
        )
        store_neurons_in_bucket(layer.hash_table, layer.neurons)
        push!(network_layers, layer)
    end
    return SlideNetwork(network_layers)
end


function train!(
    training_batches::Vector{Batch},
    n_iters::Int,
    network::SlideNetwork,
    optimizer::Optimizer,
)
    output = nothing
    for i = 1:n_iters
        loss = 0
        output = Array{Float}(undef, length(network.layers[end].neurons), 0)
        for (x_batch, y_batch) in training_batches
            y_batch_pred, last_layer_activated_neuron_ids = forward!(x_batch, network)
            output = hcat(output, y_batch_pred)
            loss += negative_sparse_logit_cross_entropy(
                y_batch_pred,
                y_batch,
                last_layer_activated_neuron_ids,
            )
            backward!(x_batch, y_batch_pred, network)
            update_weight!(network, optimizer)
            empty_neurons_attributes!(network)
        end
        println("Iteration $i, Loss $(loss / length(training_batches))")
    end
    return output
end
