using JSON
using BenchmarkTools
using Random

using Slide
using Slide.Network
using Slide.LshSimHashWrapper: LshSimHashParams, get_simhash_params
using Slide.Hash: LshParams
using Slide.FluxTraining

if (abspath(PROGRAM_FILE) == @__FILE__) || isinteractive()

    use_real_dataset = true

    # Building parameters configuration

    config_dict = JSON.parsefile("examples/slide_config.json")
    config = NamedTuple{Tuple(Symbol.(keys(config_dict)))}(values(config_dict))

    if use_real_dataset
        dataset_config = JSON.parsefile("./examples/configs/default_delicious.json")
        dataset_config["name"] *= "_" * randstring(8)
        println("Name: $(dataset_config["name"])")

        input_dim = dataset_config["n_features"]
        output_dim = dataset_config["n_classes"]
        batch_size = dataset_config["batch_size"]
        n_neurons_per_layer = [128, output_dim]

        train_loader, test_set = get_dataloaders(dataset_config)

    else
        input_dim = config.input_dim
        output_dim = config.n_neurons_per_layer[end]
        batch_size = 128
        n_neurons_per_layer = config.n_neurons_per_layer

        drop_last = false
        const N_ROWS = 4096

        x = rand(Float, config.input_dim, N_ROWS)
        y = Vector{Float}(rand(1:output_dim, N_ROWS))

        y_cat = one_hot(y)
        y_cat ./= sum(y_cat, dims = 1)
        train_loader = batch_input(x, y_cat, batch_size, drop_last)
    end

    common_lsh = LshParams(
        n_buckets = config.n_buckets,
        n_tables = config.n_tables,
        max_bucket_len = config.max_bucket_len,
    )

    lsh_params = get_simhash_params(
        common_lsh,
        convert(Vector{Int}, n_neurons_per_layer);
        signature_len = config.simhash["signature_len"],
        sample_ratio = config.simhash["sample_ratio"],
        input_size = input_dim,
    )

    network_params = Dict(
        "n_layers" => 2,
        "n_neurons_per_layer" => n_neurons_per_layer,
        "layer_activations" => Vector{String}(config.layer_activations),
        "input_dim" => input_dim,
        "lsh_params" => lsh_params,
    )


    # Data processing and training loop
    println("Data loaded, building network..........")

    network = build_network(network_params, batch_size)

    learning_rate = 0.01
    optimizer = AdamOptimizer(eta = learning_rate)

    train!(train_loader, network, optimizer; n_iters = 20, use_all_true_labels = true)
    println("DONE \n")

end
