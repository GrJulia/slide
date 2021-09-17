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

    get_output(last_layer)
end

function predict_class(
    x,
    y_true,
    network::SlideNetwork,
    topk::Int = 1;
    executor = ThreadedEx(),
)
    interference_network = inference_mode(network)
    y_pred = forward!(interference_network, x; y_true = nothing)

    topk_argmax(x) =
        if topk > 1
            partialsortperm(x, 1:topk, rev = true)
        else
            findmax(x)[2]
        end

    mapslices(topk_argmax, y_pred, dims = 1)
end
