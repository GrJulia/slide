using FLoops: @floop, ThreadedEx

using Slide.Network.Layers: forward_single_sample!, get_output


function forward!(
    x::Array{Float},
    network::SlideNetwork;
    y_true::Union{Nothing,Array{Float}} = nothing,
    executor = ThreadedEx(),
)
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

    get_output(last_layer)
end

function predict_class(
    x::Array{Float},
    y_true::Array{Float},
    network::SlideNetwork,
    topk::Int = 1
)
    interference_network = to_inference(network)
    _, y_pred = forward!(x, interference_network; y_true = nothing)

    topk_argmax(x) = if topk > 1
        partialsortperm(x, 1:topk, rev = true)
    else
        findmax(x)[2]
    end
    map(topk_argmax, y_pred)
end
