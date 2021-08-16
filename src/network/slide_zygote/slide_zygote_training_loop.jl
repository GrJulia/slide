using Zygote
using LinearAlgebra
using Flux
using Flux.Losses: logitcrossentropy

relu_scalar(x) = max(0, x)

function layer_forward_and_backward(current_activated_neurons, activated_neuron_ids, current_input, layer_activation, x_index)
    current_n_neurons = length(current_activated_neurons)
    layer_output = zeros(Float, current_n_neurons)
    for (i, neuron) in enumerate(current_activated_neurons)
        layer_output_i, neurons_gradients = withjacobian(
            (weight, bias) -> relu_scalar(dot(current_input, weight) + bias), 
            view(neuron.weight, activated_neuron_ids),neuron.bias
            )
        layer_output[i] = layer_output_i[1]
        neuron.grad_output_w[x_index] = neurons_gradients[1][1, :]
        neuron.grad_output_b[x_index] = neurons_gradients[2][1]

    end
    return layer_output
end


function forward_single_sample_zygote(
    x::SubArray{Float},
    network::SlideNetwork,
    x_index::Int;
    y_true::Union{Nothing,SubArray{Float}} = nothing,
)
    current_input = x
    activated_neuron_ids = 1:length(x)

    for (layer_idx, layer) in enumerate(network.layers)

        dense_input = zeros(Float, length(layer.neurons[1].weight))
        dense_input[activated_neuron_ids] = current_input

        min_sampling_threshold = layer.hash_tables.min_threshold
        sampling_ratio = layer.hash_tables.sampling_ratio

        # Get activated neurons
        current_activated_neuron_ids = collect(
            retrieve(
                layer.hash_tables.lsh,
                @view dense_input[:];
                threshold = max(
                    min_sampling_threshold,
                    floor(Int, length(layer.neurons) * sampling_ratio),
                ),
            ),
        )

        if !(isnothing(y_true)) && (layer_idx == length(network.layers))
            union!(current_activated_neuron_ids, findall(>(0), y_true))
        end

        #mark_ids!(layer.hash_tables, current_activated_neuron_ids)

        layer.active_neuron_ids[x_index] = current_activated_neuron_ids

        current_input = layer_forward_and_backward(
            layer.neurons[current_activated_neuron_ids],
            activated_neuron_ids, 
            current_input, 
            layer.layer_activation,
            x_index
        )

        layer.output[x_index] = current_input
        activated_neuron_ids = current_activated_neuron_ids
    end
end


function forward_zygote!(
    x::Array{Float},
    network::SlideNetwork;
    y_true::Union{Nothing,Array{Float}} = nothing,
)::Tuple{Vector{Vector{Float}},Vector{Vector{Id}}}
    batch_size = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    last_layer = network.layers[end]

    for layer in network.layers
        new_batch!(layer, batch_size)
    end

    @views for i = 1:batch_size
        forward_single_sample_zygote(
            x[:, i],
            network,
            i;
            y_true = isnothing(y_true) ? y_true : y_true[:, i],
        )
    end

    last_layer.output, last_layer.active_neuron_ids
end

function handle_batch_backward_zygote!(
    x,
    y,
    y_true,
    network::SlideNetwork,
    i::Int,
)
    da = Zygote.gradient(output -> logitcrossentropy(y_true, softmax(output)), y)[1] #dl_dypred
    @inbounds for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = layer.active_neuron_ids[i]

        if l == 1
            previous_neurons = Vector{Id}(1:length(x))
        else
            previous_neurons = network.layers[l-1].active_neuron_ids[i]
        end
        for (k, neuron) in enumerate(view(layer.neurons, active_neurons))
            neuron.is_active = true
            
            if l == length(network.layers)
                neuron.weight_gradients[previous_neurons] += neuron.grad_output_w[i] * da[k]
                neuron.bias_gradients[i] += neuron.grad_output_b[i] * da[k]
            else
                break
            end
        end
    end
end


function backward_zygote!(
    x::Matrix{Float},
    y_pred::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
)
    n_samples = size(x)[2]
    for i = 1:n_samples
        handle_batch_backward_zygote!(x[:, i], y_pred[i], y_true[i], network, i)
    end
end


function train_zygote!(
    training_batches,
    network::SlideNetwork,
    optimizer::Optimizer ;
    n_iters::Int,
    use_all_true_labels::Bool = true
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
