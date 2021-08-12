using Zygote
using Flux
using LinearAlgebra

const Float = Float64

function forward_sample(x, w, b)
    transpose(w) * x .+ b
end

function l2_norm(y_pred, y_true)
    return sum((y_true - y_pred) .^ 2)
end


mutable struct CustomLayer
    w
    b
    activation
    grad_output_w
    grad_output_b
    grad_w
    grad_b
end

function CustomLayer(in_dim, out_dim, activation)
    return CustomLayer(randn(in_dim, out_dim), randn(), activation, nothing, nothing, nothing, nothing)
end

function forward_and_grad(x, w, b, activation)
    output, grads = withjacobian((weight, bias) -> activation.(forward_sample(x, weight, bias)), w, b)
    output, grads
end

n_layers = 2
input_dim = 8
output_dim = 2

layers = [
    CustomLayer(input_dim, 4, relu),
    CustomLayer(4, output_dim, relu),
]

y_true = [1.0, 0.0]
x = randn(input_dim)

function training_loop(x, layers, y_true)
    current_input = x
    for layer in layers
        output, grads = forward_and_grad(current_input, layer.w, layer.b, layer.activation)
        layer.grad_output_w = grads[1]
        layer.grad_output_b = grads[2]
        current_input = output
    end
    da = gradient((output) -> l2_norm(output, y_true), current_input)[1]
    for i = length(layers):-1:1
        layer = layers[i]
        if i == length(layers)
            layer.grad_w = reshape(layer.grad_output_w' * da, size(layer.w))
            layer.grad_b = dot(layer.grad_output_b, da)
        else
            da = layers[i + 1].w * da
            layer.grad_w = reshape(layer.grad_output_w' * da, size(layer.w))
            layer.grad_b = dot(layer.grad_output_b, da)
        end

    end
    return current_input
end

output = training_loop(x, layers, y_true)




