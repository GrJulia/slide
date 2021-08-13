using Slide.Network: Batch, Float, extract_weights_and_ids
using Slide.Network.HashTables: update!
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
        activation_name_to_function[layer_activation];
        batch_size = batch_size,
    )
end

function train!(
    training_batches,
    test_set,
    network::SlideNetwork,
    optimizer::Optimizer,
    logger::Logger;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(15),
    use_all_true_labels::Bool = true,
    test_parameters::Dict,
) where {S<:AbstractScheduler}
    for i = 1:n_iters
        loss = 0
        for (n, (x_batch, y_batch)) in enumerate(training_batches)
            println("Iteration $i , batch $n")
            step!(logger)

            time_stats = @timed begin
                y_batch_or_nothing = if use_all_true_labels
                    y_batch
                else
                    nothing
                end

                forward_stats =
                    @timed forward!(x_batch, network; y_true = y_batch_or_nothing)
                y_batch_pred, last_layer_activated_neuron_ids = forward_stats.value
                log_scalar!(logger, "forward_time", forward_stats.time)

                println("Forward time $(forward_stats.time)")

                y_batch_activated = [
                    view(y_batch, last_layer_activated_neuron_ids[i], i) for
                    i = 1:size(y_batch, 2)
                ]

                batch_loss, saved_softmax =
                    negative_sparse_logit_cross_entropy(y_batch_pred, y_batch_activated)
                loss += batch_loss

                backward_stats = @timed backward!(
                    x_batch,
                    y_batch_pred,
                    y_batch_activated,
                    network,
                    saved_softmax,
                )

                log_scalar!(logger, "backward_time", backward_stats.time)
                println("Backward time $(backward_stats.time)")

                update_stats = @timed begin
                    update_weight!(network, optimizer)
                    zero_neuron_attributes!(network)
                end

                log_scalar!(logger, "update_time", update_stats.time)
                println("Update time $(update_stats.time)")
            end

            println("Training step done in $(time_stats.time)")
            log_scalar!(logger, "train_step time", time_stats.time)

            if n % test_parameters["test_frequency"] == 0
                test_accuracy = compute_accuracy(
                    network,
                    test_set,
                    test_parameters["n_test_batches"],
                    test_parameters["topk"],
                )
                log_scalar!(logger, "test_acc", test_accuracy)
            end

            scheduler(n) do
                for layer in network.layers
                    htable_update_stats = @timed update!(
                        layer.hash_tables,
                        extract_weights_and_ids(layer.neurons),
                    )

                    println("Hashtable $(layer.id) updated in $(htable_update_stats.time)")
                    log_scalar!(logger, "hashtable_$(layer.id)", ht_update_stats.time)
                end
                optimizer_end_epoch_step!(optimizer)
            end
        end

        println("Iteration $i, Loss $(loss / length(training_batches))")
        log_scalar!(logger, "train_loss", loss / length(training_batches))
    end
end
