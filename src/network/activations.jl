sigmoid(x) = 1 ./ (1 .+ exp.(.-x))

identity(x) = x

function sparse_softmax(x)
    λ = maximum(filter(a -> a != 0, x))
    sparse_exp_x = map(a -> Int(a != 0) * exp(a - λ), x)
    return sparse_exp_x ./ sum(sparse_exp_x)
end

activation_name_to_function =
    Dict("identity" => identity, "sparse_softmax" => sparse_softmax, "sigmoid" => sigmoid)
