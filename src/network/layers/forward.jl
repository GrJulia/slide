using LinearAlgebra: dot

using Slide.LSH: retrieve


const SlideOutput = Tuple{Vector{Float},Vector{Id}}

function forward_single_sample!(
    layer::SlideLayer{A,F,H,O},
    input::U,
    x_index::Int,
    ::Nothing,
)::SlideOutput where {A,F,H,O,U<:AbstractVector{Float}}
    current_activated_neuron_ids = _get_active_ids(layer, input, Id[])

    _forward!(layer, input, current_activated_neuron_ids, :, x_index)
end

function forward_single_sample!(
    layer::SlideLayer{A,F,H,O},
    input::U,
    x_index::Int,
    y_true::Vector{Id},
)::SlideOutput where {A,F,H,O,U<:AbstractVector{Float}}
    current_activated_neuron_ids = _get_active_ids(layer, input, y_true)

    _forward!(layer, input, current_activated_neuron_ids, :, x_index)
end

@inline function forward_single_sample!(
    layer::SlideLayer{A,F,H,O},
    input::Tuple{U,P},
    x_index::Int,
    ::Nothing,
)::SlideOutput where {A,F,H,O,U<:AbstractVector{Float},P<:AbstractVector{Id}}
    forward_single_sample!(layer, input, x_index, Id[])
end

function forward_single_sample!(
    layer::SlideLayer{A,F,H,O},
    input::Tuple{U,P},
    x_index::Int,
    y_true::Vector{Id},
)::SlideOutput where {A,F,H,O,U<:AbstractVector{Float},P<:AbstractVector{Id}}
    current_input, activated_neuron_ids = input

    dense_input = zeros(Float, size(layer.weights, 1))
    dense_input[activated_neuron_ids] = current_input

    current_activated_neuron_ids = _get_active_ids(layer, dense_input, y_true)

    _forward!(
        layer,
        current_input,
        current_activated_neuron_ids,
        activated_neuron_ids,
        x_index,
    )
end

function _get_active_ids(layer, dense_input, y_true)
    min_sampling_threshold = layer.hash_tables.min_threshold
    sampling_ratio = layer.hash_tables.sampling_ratio

    current_activated_neuron_ids = collect(
        retrieve(
            layer.hash_tables.lsh,
            @view dense_input[:];
            threshold = max(
                min_sampling_threshold,
                floor(Int, size(layer.weights, 2) * sampling_ratio),
            ),
        ),
    )
    union!(current_activated_neuron_ids, y_true)
    current_activated_neuron_ids
end

function _forward!(
    layer,
    sparse_input,
    current_activated_neuron_ids,
    activated_neuron_ids,
    x_index,
)
    layer_output = layer.biases[current_activated_neuron_ids]

    @views for (i, id) in enumerate(current_activated_neuron_ids)
        layer_output[i] += dot(sparse_input, layer.weights[activated_neuron_ids, id])
    end

    layer.active_neuron_ids[x_index] = current_activated_neuron_ids
    layer.output[x_index] = layer.layer_activation(layer_output)

    layer.output[x_index], layer.active_neuron_ids[x_index]
end
