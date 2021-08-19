using Slide.Network
using Slide.ZygoteNetwork
using Slide.FluxTraining
using Slide.Network: compute_accuracy, forward!
using Slide.Network.Optimizers
using Slide.Network.Layers: SlideLayer, extract_weights_and_ids
using Slide.Network.Optimizers: AbstractOptimizer, AdamAttributes, optimizer_end_epoch_step!
using Slide.Network.HashTables: update!

function train_zygote!(
    training_batches,
    test_set,
    network::SlideNetwork,
    optimizer::Opt,
    logger::Logger;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(15),
    use_all_true_labels::Bool = true,
    test_parameters::Dict,
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
                log_scalar!(logger, "forward_time", forward_stats.time)

                println("Forward time $(forward_stats.time)")

                y_batch_activated = [
                    view(y_batch, last_layer_activated_neuron_ids[i], i) for
                    i = 1:size(y_batch, 2)
                ]

                backward_stats = @timed backward_zygote!(
                    x_batch,
                    y_batch_pred,
                    y_batch_activated,
                    network,
                )

                log_scalar!(logger, "backward_time", backward_stats.time)
                println("Backward time $(backward_stats.time)")

                update_stats = @timed update_weight!(network, optimizer)


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
                        extract_weights_and_ids(layer.weights),
                    )

                    println("Hashtable $(layer.id) updated in $(htable_update_stats.time)")
                    log_scalar!(logger, "hashtable_$(layer.id)", htable_update_stats.time)
                end
                optimizer_end_epoch_step!(optimizer)
            end
        end

        println("Iteration $i, Loss $(loss / length(training_batches))")
        log_scalar!(logger, "train_loss", loss / length(training_batches))
    end
end
