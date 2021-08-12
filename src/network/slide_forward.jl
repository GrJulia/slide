using LinearAlgebra: dot
using FLoops: @floop, ThreadedEx

using Slide.LSH: retrieve


function forward_single_sample(
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
                    length(layer.neurons) ÷ sampling_ratio,
                ),
            ),
        )

        if !(isnothing(y_true)) && (layer_idx == length(network.layers))
            union!(current_activated_neuron_ids, findall(>(0), y_true))
        end

        layer.active_neurons[x_index] = current_activated_neuron_ids

        current_n_neurons = length(current_activated_neuron_ids)
        layer_output = zeros(Float, current_n_neurons)

        for (i, neuron) in enumerate(@view layer.neurons[current_activated_neuron_ids])
            layer_output[i] =
                dot(current_input, view(neuron.weight, activated_neuron_ids)) + neuron.bias
        end

        layer_activation = layer.layer_activation
        current_input = layer_activation(layer_output)

        layer.output[x_index] = current_input
        activated_neuron_ids = current_activated_neuron_ids
    end
end

function forward!(
    x::Array{Float},
    network::SlideNetwork;
    y_true::Union{Nothing,Array{Float}} = nothing,
    executor = ThreadedEx(),
)::Tuple{Vector{Vector{Float}},Vector{Vector{Id}}}
    n_samples = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    last_layer = network.layers[end]

    @views @floop executor for i = 1:n_samples
        forward_single_sample(
            x[:, i],
            network,
            i;
            y_true = isnothing(y_true) ? y_true : y_true[:, i],
        )
    end

    last_layer.output, last_layer.active_neurons
end

function predict_class(
    x::Array{Float},
    y_true::Array{Float},
    network::SlideNetwork,
    topk::Int = 1;
    executor = ThreadedEx(),
)
    y_active_pred, active_ids = forward!(x, network; y_true)

    y_pred = zeros(Float, size(y_true))

    @floop executor for i = 1:length(active_ids)
        ids = active_ids[i]
        y_pred[ids, i] = y_active_pred[i]
    end

    topk_argmax(x) = partialsortperm(x, 1:topk, rev = true)
    return mapslices(topk_argmax, y_pred, dims = 1)
end
