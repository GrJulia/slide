sigmoid(x) = 1 ./ (1 .+ exp.(.-x))

identity(x) = x

function sparse_softmax(x)
    λ = maximum(filter(a -> a != 0, x))
    sparse_exp_x = map(a -> Int(a != 0) * exp(a - λ), x)
    return sparse_exp_x ./ sum(sparse_exp_x)
end

relu(x) = max.(0, x)

activation_name_to_function = Dict(
    "identity" => identity,
    "sparse_softmax" => sparse_softmax,
    "sigmoid" => sigmoid,
    "relu" => relu,
)

@inline function gradient(::Type{typeof(sigmoid)}, x)
    return x * (1 - x)
end

@inline function gradient(::Type{typeof(sparse_softmax)}, x, output)
    return x * (1 - x)
end

@inline function gradient(::Type{typeof(relu)}, x)
    return Int(x > 0)
end
