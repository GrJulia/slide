using LinearAlgebra: dot
using FLoops: ThreadedEx, @floop
using SparseArrays: spzeros
using Slide: Float, SparseFloatArray
using Slide.LSH: retrieve
using Slide.Logger: log_dot_product_metrics


function (slide::SlideLayer{A,F,H,O})(
    x::T;
    true_labels::Union{Nothing,<:AbstractMatrix{Float}} = nothing,
    executor = ThreadedEx(),
) where {A,F,H,O,T<:AbstractMatrix{Float}}
    batch_size = size(x, 2)

    output = Vector{SlideOutput}(undef, batch_size)

    true_labels = if isnothing(true_labels)
        spzeros(Id, 1, batch_size)
    else
        true_labels
    end

    @views @floop executor for sample_id = 1:batch_size
        input = x[:, sample_id]
        output[sample_id] = forward_single_sample!(
            slide,
            input,
            sample_id,
            findall(>(0), true_labels[:, sample_id]),
        )
    end

    output
end

function forward_single_sample!(
    layer::SlideLayer{A,F,H,O},
    input::U,
    x_index::Int,
    ::Nothing,
)::SlideOutput where {A,F,H,O,U<:AbstractVector{Float}}
    current_active_neuron_ids = _get_active_ids(layer, input, Id[])

    _forward!(layer, input, current_active_neuron_ids, :, x_index)
end

function forward_single_sample!(
    layer::SlideLayer{A,F,H,O},
    input::U,
    x_index::Int,
    y_true::Vector{Id},
)::SlideOutput where {A,F,H,O,U<:AbstractVector{Float}}
    current_active_neuron_ids = _get_active_ids(layer, input, y_true)

    _forward!(layer, input, current_active_neuron_ids, :, x_index)
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

    current_active_neuron_ids = _get_active_ids(layer, dense_input, y_true)

    _forward!(
        layer,
        current_input,
        current_active_neuron_ids,
        activated_neuron_ids,
        x_index,
    )
end

function _get_active_ids(layer, dense_input, y_true)
    htables = layer.hash_tables

    max_neurons = max(
        htables.min_threshold,
        floor(Int, size(layer.weights, 2) * htables.sampling_ratio),
    )
    current_active_neuron_ids =
        collect(retrieve(htables.lsh, @view (dense_input[:]); threshold = max_neurons))
    union!(current_active_neuron_ids, y_true)

    current_active_neuron_ids
end

function _forward!(
    layer,
    sparse_input,
    current_active_neuron_ids,
    activated_neuron_ids,
    x_index,
)
    layer_output = layer.bias[current_active_neuron_ids]

    @debug begin
        if x_index == 1
            log_dot_product_metrics(
                layer.id,
                sparse_input,
                layer.weights[activated_neuron_ids, :],
                current_active_neuron_ids,
            )
        end
    end

    @views for (i, id) in enumerate(current_active_neuron_ids)
        layer_output[i] += dot(sparse_input, layer.weights[activated_neuron_ids, id])
    end

    layer.active_neuron_ids[x_index] = current_active_neuron_ids
    layer.output[x_index] = layer.activation(layer_output)

    layer.output[x_index], layer.active_neuron_ids[x_index]
end
