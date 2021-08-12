using Slide.Network: Batch, Float
using Slide.Hash: AbstractLshParams
using Slide.FluxTraining: Logger, log_scalar!, step!
using Zygote

function forward_single_sample_zygote!(
    x,
    network::SlideNetwork,
    activated_neuron_ids,
    activated_neurons_weights,
    activated_neurons_bias,
)
    n_layers = length(network.layers)
    current_input = x
    for i = 1:n_layers
        layer = network.layers[i]
        current_n_neurons = length(layer.neurons)
        layer_activation = layer.layer_activation
        layer_output = Vector{Float}()

        for (k, neuron_id) in enumerate(activated_neuron_ids[i])
            current_weight = activated_neurons_weights[i][k]
            current_bias = activated_neurons_bias[i][k]
            current_layer_output =
                dot(current_input, current_weight) + current_bias
            push!(layer_output, current_layer_output)
        end
        current_input = layer_activation(layer_output)
    end
    return current_input
end


function train_zygote!(
    training_batches,
    network::SlideNetwork ;
    n_iters::Int,
)
    for i = 1:n_iters
        for (x_batch, y_batch) in training_batches
            for k in 1:size(x_batch)[2]
                #activated_neuron_ids = [[id for id in retrieve(layer.hash_tables.lsh, (@view x_batch[:, k]))] for layer in network.layers]
                activated_neuron_ids = [rand(1:length(layer.neurons), 5) for layer in network.layers]
                activated_neurons_weights = [[network.layers[i].neurons[id].weight for id in activated_neuron_ids[i]] for i in 1:length(activated_neuron_ids)]
                activated_neurons_bias = [[network.layers[i].neurons[id].bias for id in activated_neuron_ids[i]] for i in 1:length(activated_neuron_ids)]
                output = forward_single_sample_zygote!(x_batch[:, k], network, activated_neuron_ids, activated_neurons_weights, activated_neurons_bias)
                loss = negative_cross_entropy_zygote(output, y_batch)
                x_sample = x_batch[:, k]
                grads_w, grads_b = Zygote.gradient((weight, bias) -> negative_cross_entropy_zygote(forward_single_sample_zygote!(x_sample, network, activated_neuron_ids, weight, bias), y_batch),
                    activated_neurons_weights, activated_neurons_bias
                )
            end
        end
    end
end

function negative_cross_entropy_zygote(
    output,
    y_true,
)
    λ, argmax_output = findmax(output)
    sparse_exp_output = map(a -> exp(a - λ), output)
    return  - sum(
        y_true .* (
            (output .- λ) .- log1p(
                sum(
                    i == argmax_output ? 0.0 : sparse_exp_output[i] for
                    i = 1:length(sparse_exp_output)
                ),
            )
        ),
    )
end