function build_and_train(
    x::Matrix{Float32},
    y::Vector{Int},
    n_iters::Int,
    batch_size::Int,
    drop_last::Bool,
    network_params::Dict,
    learning_rate::Float64,
)::Matrix{Float64}
    network = build_network(
        network_params["n_layers"],
        network_params["n_neurons_per_layer"],
        network_params["layer_activations"],
        network_params["input_dim"],
        network_params["hash_tables"],
        batch_size,
    )
    y_cat = one_hot(y)
    batches = batch_input(x, y_cat, batch_size, drop_last)
    output = nothing
    for i = 1:n_iters
        loss = 0
        output = Array{typeof(x[1])}(undef, length(network.layers[end].neurons), 0)
        for (x_batch, y_batch) in batches
            y_batch_pred = forward(x_batch, network)
            output = hcat(output, y_batch_pred)
            loss += cross_entropy(y_batch_pred, y_batch)
            backward!(x_batch, y_batch_pred, network)
            update_weight!(network, learning_rate)
            empty_neurons_attributes!(network)
        end
        println("Iteration $i, Loss $(loss / length(batches))")
    end
    return output
end
