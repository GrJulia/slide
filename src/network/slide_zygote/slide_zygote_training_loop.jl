using Zygote
using LinearAlgebra
using Flux

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
        neuron.grad_output_w[x_index] = neurons_gradients[1]
        neuron.grad_output_b[x_index] = neurons_gradients[2]

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
        # Get activated neurons and mark them as changed
        current_activated_neuron_ids = collect(
            retrieve(
                layer.hash_tables.lsh,
                @view dense_input[:];
                threshold = max(
                    min_sampling_threshold,
                    length(layer.neurons) ÷ sampling_ratio,
                ),
            ),
        )

        if !(isnothing(y_true)) && (layer_idx == length(network.layers))
            union!(current_activated_neuron_ids, findall(>(0), y_true))
        end

        mark_ids!(layer.hash_tables, current_activated_neuron_ids)

        layer.active_neurons[x_index] = current_activated_neuron_ids

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
    n_samples = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    last_layer = network.layers[end]

    @views for i = 1:n_samples
        forward_single_sample_zygote(
            x[:, i],
            network,
            i;
            y_true = isnothing(y_true) ? y_true : y_true[:, i],
        )
    end

    last_layer.output, last_layer.active_neurons
end