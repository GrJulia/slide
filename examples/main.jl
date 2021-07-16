using JSON
using BenchmarkTools
using Slide.Network


function main(
    x::Matrix{Float32},
    y::Vector{Int},
    n_iters::Int,
    batch_size::Int,
    drop_last::Bool,
    network_params::Dict,
)::Matrix{Float32}
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
    for _ = 1:n_iters
        output = Array{typeof(x[1])}(undef, length(network.layers[end].neurons), 0)
        for (x_batch, y_batch) in batches
            y_batch_pred = forward(x_batch, network)
            output = hcat(output, y_batch_pred)
            loss = cross_entropy(y_batch_pred, y_batch)
            backward!(x_batch, y_batch_pred, y_batch, loss, network)
        end
    end
    output
end

function build_random_configuration()
    n_layers = rand(1:10)
    input_dim = rand(8:32)
    n_neurons_per_layer = [rand(1:10) for _ = 1:n_layers]
    layer_activations = ["sigmoid" for _ = 1:n_layers-1]
    push!(layers_activations, "sparse_softmax")
    n_buckets = 2
    return (
        n_layers = n_layers,
        n_neurons_per_layer = n_neurons_per_layer,
        layer_activations = layer_activations,
        input_dim = input_dim,
        n_buckets = n_buckets,
    )
end

if (abspath(PROGRAM_FILE) == @__FILE__) || isinteractive()
    random_config = false
    benchmark = false
    if random_config
        config = build_random_configuration()
    else
        config_dict = JSON.parsefile("examples/slide_config.json")
        config = NamedTuple{Tuple(Symbol.(keys(config_dict)))}(values(config_dict))
    end

    hash_tables = [HashTable([[] for _ = 1:config.n_buckets]) for _ = 1:config.n_layers]
    network_params = Dict(
        "n_layers" => config.n_layers,
        "n_neurons_per_layer" => Vector{Int}(config.n_neurons_per_layer),
        "layer_activations" => Vector{String}(config.layer_activations),
        "input_dim" => config.input_dim,
        "hash_tables" => hash_tables,
    )
    output_dim = config.n_neurons_per_layer[end]

    x = rand(Float32, config.input_dim, 4096)
    y = rand(1:output_dim, 4096)
    output = main(x, y, 1, 256, false, network_params)
    println("DONE \n")
end
