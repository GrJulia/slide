using JSON
using BenchmarkTools
using Slide.Network


function main(
    x::Matrix{Float32},
    n_iters::Int64,
    batch_size::Int64,
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
    batches = batch_input(x, batch_size, drop_last)
    output = nothing
    for _ = 1:n_iters
        output = Array{typeof(x[1])}(undef, length(network.layers[end].neurons), 0)
        for batch in batches
            output = hcat(output, forward(batch, network))
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
        config_dict = JSON.parsefile("slide_config.json")
        config = NamedTuple{Tuple(Symbol.(keys(config_dict)))}(values(config_dict))
    end

    hash_tables = [HashTable([[] for _ = 1:config.n_buckets]) for _ = 1:config.n_layers]
    network_params = Dict(
        "n_layers" => config.n_layers,
        "n_neurons_per_layer" => Vector{Int64}(config.n_neurons_per_layer),
        "layer_activations" => Vector{String}(config.layer_activations),
        "input_dim" => config.input_dim,
        "hash_tables" => hash_tables,
    )

    if benchmark
        rows_values = [64, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768]
        timings = []
        for _ = 1:2
            main(rand(Float32, config.input_dim, 64), 1, 32, false, network_params)
        end
        for i = 1:length(rows_values)
            x = rand(Float32, config.input_dim, rows_values[i])
            append!(timings, @elapsed main(x, 1, 32, false, network_params))
        end
    else
        output = main(rand(Float32, config.input_dim, 4096), 1, 256, false, network_params)
    end
    println("DONE \n")
end
