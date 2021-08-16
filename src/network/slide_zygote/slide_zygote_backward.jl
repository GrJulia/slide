using Zygote
using LinearAlgebra
using Flux
using Flux.Losses: logitcrossentropy
using FLoops: @floop, ThreadedEx


function handle_batch_backward_zygote!(
    x::T,
    y::U,
    y_true::P,
    network::SlideNetwork,
    i::Int,
) where {T<:FloatVector,P<:FloatVector,U<:FloatVector}
    da = Zygote.gradient(output -> logitcrossentropy(y_true, softmax(output)), y)[1]
    @inbounds for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = layer.active_neuron_ids[i]

        if l == 1
            previous_neurons = Vector{Id}(1:length(x))
        else
            previous_neurons = network.layers[l-1].active_neuron_ids[i]
        end

        if l < length(network.layers)
            da = [
                sum([
                    da[j] * network.layers[l+1].neurons[j].weight[k] for j = 1:length(da)
                ]) for k in active_neurons
            ]
        else

        end

        for (k, neuron) in enumerate(view(layer.neurons, active_neurons))
            neuron.is_active = true
            @views axpy!(da[k], neuron.grad_output_w[i], neuron.weight_gradients[previous_neurons])
            neuron.bias_gradients[i] += neuron.grad_output_b[i] * da[k]
        end
    end
end


function backward_zygote!(
    x::Matrix{Float},
    y_pred::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
    executor = ThreadedEx(),
)
    n_samples = size(x)[2]
    @views @floop executor for i = 1:n_samples
        handle_batch_backward_zygote!(x[:, i], y_pred[i], y_true[i], network, i)
    end
end
