using Slide.Network: Batch, Float
using Slide.Network.HashTables: update!
using Slide.Hash: AbstractLshParams
using Slide.SlideLogger: Logger, step!
using Slide.Network.Layers: SlideLayer, extract_weights_and_ids
using Slide.Network.Optimizers: AbstractOptimizer, AdamAttributes, optimizer_end_epoch_step!


function build_network(network_params::Dict)::SlideNetwork
    network_layers = Vector{SlideLayer}()

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
            layer_id,
            network_params["layer_activations"][layer_id],
            network_params["lsh_params"][layer_id],
        )
        push!(network_layers, layer)
    end

    SlideNetwork(network_layers)
end

function build_layer(
    input_dim,
    output_dim,
    layer_id,
    layer_activation,
    lsh_params::T,
) where {T<:AbstractLshParams}
    SlideLayer(
        layer_id,
        input_dim,
        output_dim,
        lsh_params,
        activation_name_to_function[layer_activation],
        AdamAttributes(input_dim, output_dim),
    )
end


function train!(
    training_batches,
    test_set,
    network::SlideNetwork,
    optimizer::Opt,
    logger::Logger;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(15),
    use_all_true_labels::Bool = true,
    test_parameters::Dict,
    use_zygote::Bool = false,
) where {S<:AbstractScheduler,Opt<:AbstractOptimizer}
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
                @info "forward_time" forward_stats.time

                println("Forward time $(forward_stats.time)")

                y_batch_activated = [
                    view(y_batch, last_layer_activated_neuron_ids[i], i) for
                    i = 1:size(y_batch, 2)
                ]

                batch_loss, saved_softmax =
                    negative_sparse_logit_cross_entropy(y_batch_pred, y_batch_activated)
                loss += batch_loss

                @info "train_loss" batch_loss

                if use_zygote
                    backward_stats =
                        @timed backward_zygote!(x_batch, y_batch_activated, network)
                else
                    backward_stats = @timed backward!(
                        x_batch,
                        y_batch_pred,
                        y_batch_activated,
                        network,
                        saved_softmax,
                    )
                end

                @info "backward_time" backward_stats.time
                println("Backward time $(backward_stats.time)")

                update_stats = @timed update_weight!(network, optimizer)

                @info "update_time" update_stats.time
                println("Update time $(update_stats.time)")
            end

            println("Training step done in $(time_stats.time)")
            @info "train_step time" time_stats.time

            if n % test_parameters["test_frequency"] == 0
                test_accuracy = compute_accuracy(
                    network,
                    test_set,
                    test_parameters["n_test_batches"],
                    test_parameters["topk"],
                )
                @info "test_acc" test_accuracy
            end

            scheduler(n) do
                for layer in network.layers
                    htable_update_stats = @timed update!(
                        layer.hash_tables,
                        extract_weights_and_ids(layer.weights),
                    )

                    println("Hashtable $(layer.id) updated in $(htable_update_stats.time)")
                    @info "hashtable_$(layer.id)" htable_update_stats.time
                end
                optimizer_end_epoch_step!(optimizer)
            end
        end

        println("Iteration $i, Loss $(loss / length(training_batches))")
    end
end
