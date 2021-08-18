using Zygote
using LinearAlgebra
using Flux
using FLoops: @floop, ThreadedEx

using Slide.LSH: retrieve
using Slide.Network.Layers: new_batch!

function layer_forward_and_backward(
    current_activated_neurons,
    activated_neuron_ids,
    current_input,
    layer_activation,
    x_index,
)
    current_n_neurons = length(current_activated_neurons)
    layer_output = zeros(Float, current_n_neurons)
    for (i, neuron) in enumerate(current_activated_neurons)
        layer_output_i, neurons_gradients = withjacobian(
            (weight, bias) -> layer_activation(dot(current_input, weight) + bias),
            view(neuron.weight, activated_neuron_ids),
            neuron.bias,
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

        layer.active_neuron_ids[x_index] = current_activated_neuron_ids

        current_input = layer_forward_and_backward(
            layer.neurons[current_activated_neuron_ids],
            activated_neuron_ids,
            current_input,
            layer.layer_activation,
            x_index,
        )

        layer.output[x_index] = current_input
        activated_neuron_ids = current_activated_neuron_ids
    end
end


function forward_zygote!(
    x::Array{Float},
    network::SlideNetwork;
    y_true::Union{Nothing,Array{Float}} = nothing,
    executor = ThreadedEx(),
)::Tuple{Vector{Vector{Float}},Vector{Vector{Id}}}
    batch_size = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    last_layer = network.layers[end]

    for layer in network.layers
        new_batch!(layer, batch_size)
    end

    @views @floop executor for i = 1:batch_size
        forward_single_sample_zygote(
            x[:, i],
            network,
            i;
            y_true = isnothing(y_true) ? y_true : y_true[:, i],
        )
    end

    last_layer.output, last_layer.active_neuron_ids
end
