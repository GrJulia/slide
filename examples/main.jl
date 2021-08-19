using JSON
using Random

using Slide
using Slide.Network
using Slide.ZygoteNetwork
using Slide.LshSimHashWrapper: LshSimHashParams, get_simhash_params
using Slide.Hash: LshParams
using Slide.FluxTraining
using Slide.Network.Optimizers: AdamOptimizer

Random.seed!(1);

if (abspath(PROGRAM_FILE) == @__FILE__) || isinteractive()

    use_real_dataset = true
    with_zygote = false

    # Building parameters configuration

    config_dict = JSON.parsefile("examples/slide_config.json")
    config = NamedTuple{Tuple(Symbol.(keys(config_dict)))}(values(config_dict))
    dataset_config = JSON.parsefile("./examples/configs/default_delicious.json")

    if use_real_dataset
        dataset_config["name"] *= "_" * randstring(8)
        println("Name: $(dataset_config["name"])")

        input_dim = dataset_config["n_features"]
        output_dim = dataset_config["n_classes"]
        batch_size = dataset_config["batch_size"]
        n_neurons_per_layer = [128, output_dim]

        train_loader, test_set = get_dataloaders(dataset_config)

        test_parameters = Dict(
            "test_frequency" => dataset_config["testing"]["test_freq"],
            "n_test_batches" => dataset_config["testing"]["n_batches"],
            "topk" => dataset_config["testing"]["top_k_classes"],
        )

    else
        input_dim = config.input_dim
        output_dim = config.n_neurons_per_layer[end]
        batch_size = 128
        n_neurons_per_layer = config.n_neurons_per_layer

        drop_last = false
        const N_ROWS = 4096

        x = rand(Float, config.input_dim, N_ROWS)
        y = Vector{Float}(rand(1:output_dim, N_ROWS))

        x_test = rand(Float, config.input_dim, N_ROWS)
        y_test = Vector{Float}(rand(1:output_dim, N_ROWS))

        y_cat = one_hot(y)
        y_cat ./= sum(y_cat, dims = 1)

        y_cat_test = one_hot(y_test)
        y_cat_test ./= sum(y_cat_test, dims = 1)

        train_loader = batch_input(x, y_cat, batch_size, drop_last)
        test_set = batch_input(x_test, y_cat_test, batch_size, drop_last)

        test_parameters = Dict("test_frequency" => 2, "n_test_batches" => 2, "topk" => 1)
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

    network = build_network(network_params)

    learning_rate = 0.001
    optimizer = AdamOptimizer(eta = learning_rate)

    logger = get_logger(dataset_config)

    if with_zygote
        train_zygote!(
            train_loader,
            test_set,
            network,
            optimizer,
            logger;
            n_iters = 1,
            scheduler = PeriodicScheduler(5),
            use_all_true_labels = true,
            test_parameters = test_parameters,
        )
    else
        train!(
            train_loader,
            test_set,
            network,
            optimizer,
            logger;
            n_iters = 3,
            scheduler = PeriodicScheduler(5),
            use_all_true_labels = true,
            test_parameters = test_parameters,
        )
    end
    println("DONE \n")

    save(logger)
end
