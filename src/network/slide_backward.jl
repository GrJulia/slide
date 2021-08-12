using Statistics: mean
using Base.Threads: @threads
using LinearAlgebra.BLAS: axpy!

const FloatVector = AbstractVector{Float}

function handle_batch_backward(
    x::T,
    y::U,
    y_true::P,
    network::SlideNetwork,
    i::Int,
    saved_softmax::Vector{Float},
) where {T<:FloatVector,P<:FloatVector,U<:FloatVector}
    @inbounds for l = length(network.layers):-1:1
        layer = network.layers[l]
        active_neurons = layer.active_neurons[i]

        if l == 1
            previous_activation = x
            previous_neurons = Vector{Id}(1:length(x))
        else
            previous_activation = network.layers[l-1].output[i]
            previous_neurons = network.layers[l-1].active_neurons[i]
        end

        for (k, neuron) in enumerate(view(layer.neurons, active_neurons))
            neuron.is_active = true
            if l == length(network.layers)
                # recall that saved_softmax's length is size(active_neurons)
                # sum(y_true): to handle multiple labels
                dz = gradient(
                    typeof(negative_sparse_logit_cross_entropy),
                    y_true[k],
                    saved_softmax[k],
                    sum(y_true),
                )
            else
                # we could only sum over the active neurons in layer l+1, but
                # here, if a neuron is not active, we're just summing 0
                da = sum(
                    next_neuron.bias_gradients[i] * next_neuron.weight[neuron.id] for
                    next_neuron in view(
                        network.layers[l+1].neurons,
                        network.layers[l+1].active_neurons[i],
                    )
                )
                dz = da * gradient(typeof(layer.layer_activation), layer.output[i][k])
            end

            neuron.bias_gradients[i] = dz
            dz = dz / length(neuron.bias_gradients)
            @views axpy!(dz, previous_activation, neuron.weight_gradients[previous_neurons])
        end
    end
end

function update_weight!(network::SlideNetwork, optimizer::Optimizer)
    for layer in network.layers
        @threads for neuron in filter(n -> n.is_active, layer.neurons)
            optimizer_step!(optimizer, neuron)
        end
    end
end

function backward!(
    x::Matrix{Float},
    y_pred::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
    network::SlideNetwork,
    saved_softmax::Vector{<:FloatVector},
)
    n_samples = size(x)[2]
    @views @threads for i = 1:n_samples
        handle_batch_backward(x[:, i], y_pred[i], y_true[i], network, i, saved_softmax[i])
    end
end
