using Slide.Network: Batch, Float
using Slide.Logger: SlideLogger, step!
using Slide.Network.Layers: extract_weights_and_ids, update_htable!
using Slide.Network.Optimizers: AbstractOptimizer


function train!(
    training_batches,
    network::SlideNetwork,
    optimizer::Opt,
    logger::SlideLogger;
    n_epochs::Int,
    use_zygote::Bool = false,
    callbacks = [],
) where {Opt<:AbstractOptimizer}

    it(epoch, n) = (epoch - 1) * length(training_batches) + n

    for epoch = 1:n_epochs
        loss = 0
        for (n, (x_batch, y_batch)) in enumerate(training_batches)
            println("Epoch $epoch , batch $n")
            step!(logger)

            time_stats = @timed begin

                forward_stats = @timed forward!(network, x_batch; y_true = y_batch)
                y_batch_pred, last_layer_activated_neuron_ids = forward_stats.value

                @info "forward_time" forward_stats.time
                println("Forward time $(forward_stats.time)")

                y_batch_activated = [
                    view(y_batch, last_layer_activated_neuron_ids[i], i) for
                    i = 1:size(y_batch, 2)
                ]

                batch_loss, saved_softmax =
                    negative_sparse_logit_cross_entropy(y_batch_pred, y_batch_activated)

                println("Loss $batch_loss")
                @info "train_loss" batch_loss

                if use_zygote
                    backward_stats =
                        @timed backward_zygote!(x_batch, y_batch_activated, network)
                else
                    backward_stats = @timed backward!(
                        network,
                        x_batch,
                        y_batch_activated,
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

            iteration = it(epoch, n)
            for callback in callbacks
                callback(iteration, network)
            end
        end

        println("Epoch $epoch, Loss $(loss / length(training_batches))")
    end
end
