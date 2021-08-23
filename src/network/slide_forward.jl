using FLoops: @floop, ThreadedEx

using Slide.Network.Layers: forward_single_sample!


function forward!(
    x::Array{Float},
    network::SlideNetwork;
    y_true::Union{Nothing,Array{Float}} = nothing,
    executor = ThreadedEx(),
)::Tuple{Vector{Vector{Float}},Vector{Vector{Id}}}
    batch_size = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    last_layer = network.layers[end]

    new_batch!(network, batch_size)

    @views @floop executor for i = 1:batch_size
        input = x[:, i]
        for layer in network.layers[1:end-1]
            input = forward_single_sample!(layer, input, i, nothing)
        end

        maybe_y_true_idxs = if !isnothing(y_true)
            findall(>(0), y_true[:, i])
        else
            nothing
        end

        forward_single_sample!(last_layer, input, i, maybe_y_true_idxs)
    end

    last_layer.output, last_layer.active_neuron_ids
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
    mapslices(topk_argmax, y_pred, dims = 1)
end
