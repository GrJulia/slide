using Slide.Network: Batch, Float
using Slide.Hash: AbstractLshParams
using Slide.FluxTraining: Logger, log_scalar!, step!

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
    test_set,
    network::SlideNetwork,
    optimizer::Optimizer,
    logger::Logger;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(30),
    use_all_true_labels::Bool = true,
    test_parameters::Dict
) where {S<:AbstractScheduler}
    for i = 1:n_iters
        loss = 0
        for (n, (x_batch, y_batch)) in enumerate(training_batches)
            println("Iteration $i , batch $n")
            step!(logger)
            time_stats = @timed begin
                println("Forward")
                if use_all_true_labels
                    y_batch_pred = forward!(x_batch, network, y_batch)
                else
                    y_batch_pred = forward!(x_batch, network, nothing)
                end
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
                update_weight!(network, optimizer)
                zero_neuron_attributes!(network)
            end
            println("Training step done")

            elapsed_time = time_stats.time
            log_scalar!(logger, "train_step time", elapsed_time)
            if n % test_parameters["test_frequency"] == 0
                train_accuracy = compute_accuracy(network, training_batches, test_parameters["n_train_batches"], test_parameters["topk"])
                test_accuracy = compute_accuracy(network, test_set, test_parameters["n_test_batches"], test_parameters["topk"])
                log_scalar!(logger, "test_acc", test_accuracy)
                log_scalar!(logger, "train_acc", train_accuracy)
            end
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
