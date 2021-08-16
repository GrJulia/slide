using Slide.Network

function train_zygote!(
    training_batches,
    test_set,
    network::SlideNetwork,
    optimizer::Optimizer,
    logger::Logger;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(15),
    use_all_true_labels::Bool = true,
    test_parameters::Dict,
)
    for i = 1:n_iters
        for (n, (x_batch, y_batch)) in enumerate(training_batches)
            println("Iteration $i , batch $n")

            time_stats = @timed begin
                y_batch_or_nothing = if use_all_true_labels
                    y_batch
                else
                    nothing
                end

                forward_stats =
                    @timed forward_zygote!(x_batch, network; y_true = y_batch_or_nothing)
                y_batch_pred, last_layer_activated_neuron_ids = forward_stats.value

                println("Forward time $(forward_stats.time)")

                y_batch_activated = [
                    view(y_batch, last_layer_activated_neuron_ids[i], i) for
                    i = 1:size(y_batch, 2)
                ]

                backward_zygote!(
                    x_batch,
                    y_batch_pred,
                    y_batch_activated,
                    network,
                )

                update_stats = @timed update_weight!(network, optimizer)

                println("Update time $(update_stats.time)")
            end
            println("Training step done in $(time_stats.time)")
        
        end
        println("Iteration $i done")
    end
end
