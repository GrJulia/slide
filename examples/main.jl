using JSON
using BenchmarkTools
using Slide.Network

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
    learning_rate = 0.01

    x = rand(Float, config.input_dim, 4096) / 10
    y = rand(1:output_dim, 4096)
    output, network = build_and_train(x, y, 5, 256, false, network_params, learning_rate)
    println("DONE \n")

    layer_id = 1
    neuron_id = 1
    weight_index = 1
    x_check = x[:, 1]
    y_check = one_hot(y)[:, 1]
    for neuron_id in 1:128
        println("Neuron $neuron_id, weight grad")
        numerical_gradient_weights(network, layer_id, neuron_id, weight_index, x_check, y_check, 0.01)
    end

    for neuron_id in 1:128
        println("Neuron $neuron_id, bias grad")
        numerical_gradient_bias(network, layer_id, neuron_id, x_check, y_check, 0.01)
    end
end
