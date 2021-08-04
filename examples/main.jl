using JSON
<<<<<<< HEAD
using Random
=======
using BenchmarkTools
>>>>>>> f094fcd09c4d1eabebeda8a194713d7a56c1bda1

using Slide
using Slide.Network
using Slide.LshSimHashWrapper: LshSimHashParams, get_simhash_params
using Slide.Hash: LshParams
<<<<<<< HEAD
using Slide.FluxTraining
=======
>>>>>>> f094fcd09c4d1eabebeda8a194713d7a56c1bda1

function build_random_configuration()
    n_layers = rand(1:10)
    input_dim = rand(8:32)
    n_neurons_per_layer = [rand(1:10) for _ = 1:n_layers]
    layer_activations = ["relu" for _ = 1:n_layers-1]
    push!(layers_activations, "identity")
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

    # Building parameters configuration

    random_config = false
    benchmark = false
    if random_config
        config = build_random_configuration()
    else
        config_dict = JSON.parsefile("examples/slide_config.json")
        config = NamedTuple{Tuple(Symbol.(keys(config_dict)))}(values(config_dict))
    end

    common_lsh = LshParams(
        n_buckets = config.n_buckets,
        n_tables = config.n_tables,
        max_bucket_len = config.max_bucket_len,
    )
    lsh_params = get_simhash_params(
        common_lsh,
        convert(Vector{Int}, config.n_neurons_per_layer);
        signature_len = config.simhash["signature_len"],
        sample_ratio = config.simhash["sample_ratio"],
        input_size = config.input_dim,
    )

    # network_params = Dict(
    #     "n_layers" => config.n_layers,
    #     "n_neurons_per_layer" => Vector{Int}(config.n_neurons_per_layer),
    #     "layer_activations" => Vector{String}(config.layer_activations),
    #     "input_dim" => config.input_dim,
    #     "lsh_params" => lsh_params,
    # )
    # output_dim = config.n_neurons_per_layer[end]
    learning_rate = 0.01
    batch_size = 128
    drop_last = false

    const N_ROWS = 4096

    # x = rand(Float, config.input_dim, N_ROWS)
    # y = Vector{Float}(rand(1:output_dim, N_ROWS))

    dataset_config = JSON.parsefile("./examples/configs/default_delicious.json")
    dataset_config["name"] *= "_" * randstring(8)
    println("Name: $(dataset_config["name"])")

    train_loader, test_set = get_dataloaders(dataset_config)

    network_params = Dict(
        "n_layers" => config.n_layers,
        "n_neurons_per_layer" => Vector{Int}(config.n_neurons_per_layer),
        "layer_activations" => Vector{String}(config.layer_activations),
        "input_dim" => dataset_config["n_features"],
        "lsh_params" => lsh_params,
    )
    output_dim = dataset_config["n_classes"]

    # # Data processing and training loop
    println("Data loaded, building network..........")

    network = build_network(network_params, batch_size)

    # y_cat = one_hot(y)
    # y_cat ./= sum(y_cat, dims = 1)
    # training_batches = batch_input(x, y_cat, batch_size, drop_last)

    optimizer = AdamOptimizer(eta = learning_rate)

    train!(training_batches, network, optimizer; n_iters = 20, use_all_true_labels = true)
    println("DONE \n")

end
