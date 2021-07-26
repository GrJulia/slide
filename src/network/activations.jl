sigmoid(x::Vector{Float}) = 1 ./ (1 .+ exp.(.-x))

identity(x::Any) = x

relu(x::Vector{Float}) = max.(0, x)

function negative_sparse_logit_cross_entropy(output::Array{Float}, y_true::Array{Float}, last_layer_activated_neuron_ids) # activated neurons + refactor (cancel log / exp)
    return -mean([negative_sparse_logit_cross_entropy_sample(
        output[:, i][last_layer_activated_neuron_ids[i]], 
        y_true[:, i][last_layer_activated_neuron_ids[i]], 
        )
        for i = 1:size(output)[end]]
        )
end

function negative_sparse_logit_cross_entropy_sample(output::Vector{Float}, y_true::Vector{Float})
    λ = maximum(output)
    sparse_exp_output = map(a -> exp(a - λ), output)
    return sum(y_true .* ((output .- λ) .- log(sum(sparse_exp_output) + eps())))
end

activation_name_to_function = Dict(
    "identity" => identity,
    "sigmoid" => sigmoid,
    "relu" => relu,
)

@inline function gradient(::Type{typeof(sigmoid)}, x::Float)
    return x * (1 - x)
end

@inline function gradient(::Type{typeof(negative_sparse_logit_cross_entropy)}, x::Float, output::Float)
    return x - output
end

@inline function gradient(::Type{typeof(relu)}, x::Float)
    return Int(x > 0)
end
