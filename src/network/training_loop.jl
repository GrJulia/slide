using Slide.Network: Batch, Float
using Slide.Hash: AbstractLshParams
using Slide.Logger: SlideLogger, step!
using Slide.Network.Layers:
    AbstractLayer, SlideLayer, extract_weights_and_ids, Dense, update_htable!
using Slide.Network.Optimizers: AbstractOptimizer, AdamAttributes


function build_network(network_params::Dict)::SlideNetwork
    network_layers = Vector{AbstractLayer}()

    for layer_id = 1:network_params["n_layers"]
        if layer_id == 1
            layer_input_dim = network_params["input_dim"]
        else
            layer_input_dim = network_params["n_neurons_per_layer"][layer_id-1]
        end
        layer_output_dim = network_params["n_neurons_per_layer"][layer_id]
        layer_type = network_params["layer_types"][layer_id]

        layer = if layer_type == :slide
            build_layer(
                layer_input_dim,
                layer_output_dim,
                layer_id,
                network_params["layer_activations"][layer_id],
                network_params["lsh_params"][layer_id],
            )
        else
            build_layer(
                layer_input_dim,
                layer_output_dim,
                layer_id,
                network_params["layer_activations"][layer_id],
            )
        end
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

function build_layer(input_dim, output_dim, layer_id, layer_activation)
    Dense(
        layer_id,
        input_dim,
        output_dim,
        activation_name_to_function[layer_activation],
        AdamAttributes(input_dim, output_dim),
    )
end

function train!(
    training_batches,
    test_set,
    network::SlideNetwork,
    optimizer::Opt,
    logger::SlideLogger;
    n_iters::Int,
    scheduler::S = PeriodicScheduler(15),
    use_all_true_labels::Bool = true,
    test_parameters::Dict,
    use_zygote::Bool = false,
) where {S<:AbstractScheduler,Opt<:AbstractOptimizer}
    for i = 1:n_iters, (n, (x_batch, y_batch)) in enumerate(training_batches)
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

            println("Loss $batch_loss")
            @info "train_loss" batch_loss

            if use_zygote
                backward_stats =
                    @timed backward_zygote!(x_batch, y_batch_activated, network)
            else
                backward_stats =
                    @timed backward!(x_batch, y_batch_activated, network, saved_softmax)
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
                htable_update_stats = @timed update_htable!(layer)

                println("Hashtable $(layer.id) updated in $(htable_update_stats.time)")
                @info "hashtable_$(layer.id)" htable_update_stats.time
            end
        end
    end
end
