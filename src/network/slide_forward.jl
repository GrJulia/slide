using FLoops: @floop, ThreadedEx


function forward!(
    network::SlideNetwork,
    x::T;
    y_true::Union{Nothing,<:AbstractArray{Float}} = nothing,
    executor = ThreadedEx(),
) where {T<:AbstractMatrix{Float}}
    batch_size = typeof(x) == Vector{Float} ? 1 : size(x)[end]
    last_layer = network.layers[end]

    new_batch!(network, batch_size)

    output = x
    for layer in network.layers[1:end-1]
        output = layer(output; executor)
    end
    last_layer(output; executor, true_labels = y_true)

    last_layer.output, last_layer.active_neuron_ids
end

function predict_class(
    x,
    y_true,
    network::SlideNetwork,
    topk::Int = 1;
    executor = ThreadedEx(),
)
    interference_network = to_inference(network)
    y_pred = forward!(interference_network, x; y_true = nothing)

    y_pred = zeros(Float, size(y_true))

    @floop executor for i = 1:length(active_ids)
        ids = active_ids[i]
        y_pred[ids, i] = y_active_pred[i]
    end

    topk_argmax(x) = partialsortperm(x, 1:topk, rev = true)
    mapslices(topk_argmax, y_pred, dims = 1)
end
