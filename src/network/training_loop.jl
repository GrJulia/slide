using Slide.Network: Batch, Float
using Slide.Hash: AbstractLshParams

function build_network(network_params::Dict, batch_size::Int)::SlideNetwork
    network_layers = Vector{Layer}()
    for layer_id = 1:network_params["n_layers"]
        if layer_id == 1
            layer_input_dim = network_params["input_dim"]
        else
            layer_input_dim = network_params["n_neurons_per_layer"][layer_id-1]
        end
        layer_output_dim = network_params["n_neurons_per_layer"][layer_id]
        layer = build_layer(
            layer_input_dim,
            layer_output_dim,
            batch_size,
            layer_id,
            network_params["layer_activations"][layer_id],
            network_params["lsh_params"][layer_id],
        )
        push!(network_layers, layer)
    end
    return SlideNetwork(network_layers)
end

function build_layer(
    input_dim,
    output_dim,
    batch_size,
    layer_id,
    layer_activation,
    lsh_params::T,
) where {T<:AbstractLshParams}
    neurons = Vector{Neuron{AdamAttributes}}()
    for neuron_id = 1:output_dim
        push!(neurons, Neuron(neuron_id, batch_size, input_dim))
    end
    return Layer(
        layer_id,
        neurons,
        lsh_params,
        activation_name_to_function[layer_activation],
    )
end

function train!(
    training_batches,
    network::SlideNetwork,
    optimizer::Optimizer;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(10),
    use_all_true_labels::Bool = true,
) where {S<:AbstractScheduler}
    for i = 1:n_iters
        loss = 0
        for (n, (x_batch, y_batch)) in enumerate(training_batches)
            println("Iteration $i, batch $n")
            start_batch_time = time()
            println("Forward")
            if use_all_true_labels
                y_batch_pred = forward!(x_batch, network, y_batch)
            else
                y_batch_pred = forward!(x_batch, network, nothing)
            end
            println("Froward time $(time() - start_batch_time)")
            last_layer_activated_neuron_ids =
                get_active_neuron_ids(network, length(network.layers))
            batch_loss, saved_softmax = negative_sparse_logit_cross_entropy(
                y_batch_pred,
                y_batch,
                last_layer_activated_neuron_ids,
            )
            loss += batch_loss
            println("Backward")
            backward!(x_batch, y_batch_pred, y_batch, network, saved_softmax)
            println("Backward time $(time() - start_batch_time)")
            update_weight!(network, optimizer)
            zero_neuron_attributes!(network)
            println("Total Batch time $(time() - start_batch_time)")
        end


        println("Iteration $i, Loss $(loss / length(training_batches))")
        scheduler(i) do
            for layer in network.layers
                update!(layer.hash_tables, layer.neurons)
            end
        end
        optimizer_end_epoch_step!(optimizer)
    end
end
