using JSON
using Random
using Logging: global_logger
using SparseArrays

using Slide
using Slide.Network
using Slide.Network.Layers: Dense, SlideLayer, update_htable!, reinit_htable!
using Slide.LshSimHashWrapper: LshSimHashParams, get_simhash_params
using Slide.Hash: LshParams
using Slide.DataLoading: get_dense_dataloaders, get_sparse_datasets
using Slide.Logger: get_logger, save
using Slide.Network.Optimizers: AdamOptimizer

Random.seed!(1);

const HT_UPDATE_PERIOD = 50
const HT_REINIT_PERIOD = 1000

function hashtable_update!(network)
    for (id, layer) in enumerate(network.layers)
        htable_update_stats = @timed update_htable!(layer)

        println("Hashtable $id updated in $(htable_update_stats.time)")
        @info "hashtable_$id-update" htable_update_stats.time
    end
end

function hashtable_reinit!(network)
    for (id, layer) in enumerate(network.layers)
        htable_update_stats = @timed reinit_htable!(layer)

        println("Hashtable $id reconstructed in $(htable_update_stats.time)")
        @info "hashtable_$id-reconstructed" htable_update_stats.time
    end
end

function test_accuracy(network, test_set, n_test_batches, topk)
    test_acc = compute_accuracy(network, test_set, n_test_batches, topk)
    println("Test accuracy: $test_acc")
    @info "test_acc" test_acc
end

if (abspath(PROGRAM_FILE) == @__FILE__) || isinteractive()

    use_real_dataset = false
    use_zygote = false

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

        train_loader, test_set = get_sparse_datasets(dataset_config)

        const save_parameters = Dict(
            "save_frequency" => dataset_config["n_samples"] ÷ dataset_config["batch_size"],
            "model_path" => joinpath(dataset_config["logger"]["logging_path"], dataset_config["name"]),
        )

        const test_parameters = Dict(
            "test_frequency" => dataset_config["testing"]["test_freq"],
            "n_test_batches" => dataset_config["testing"]["n_batches"],
            "topk" => dataset_config["testing"]["top_k_classes"],
        )

    else
        dataset_config["name"] *= "_toy_" * randstring(8)

        input_dim = config.input_dim
        output_dim = config.n_neurons_per_layer[end]
        batch_size = 128
        n_neurons_per_layer = config.n_neurons_per_layer

        drop_last = false
        const N_ROWS = 4096

        x = sprand(Float, config.input_dim, N_ROWS, 0.5)
        y = Vector{Float}(rand(1:output_dim, N_ROWS))

        x_test = rand(Float, config.input_dim, N_ROWS)
        y_test = Vector{Float}(rand(1:output_dim, N_ROWS))

        y_cat = one_hot(y)
        y_cat ./= sum(y_cat, dims = 1)

        y_cat_test = one_hot(y_test)
        y_cat_test ./= sum(y_cat_test, dims = 1)

        train_loader = batch_input(x, y_cat, batch_size, drop_last)
        test_set = batch_input(x_test, y_cat_test, batch_size, drop_last)

        const save_parameters = Dict(
            "save_frequency" => 4096 ÷ 128,
            "model_path" => joinpath(dataset_config["logger"]["logging_path"], dataset_config["name"]),
        )

        const test_parameters =
            Dict("test_frequency" => 2, "n_test_batches" => 2, "topk" => 1)
    end

    # Data processing and training loop
    println("Data loaded, building network..........")

    common_lsh = LshParams(
        n_buckets = config.n_buckets,
        n_tables = config.n_tables,
        max_bucket_len = config.max_bucket_len,
    )

    lsh_params = get_simhash_params(
        common_lsh,
        convert(Vector{Int}, n_neurons_per_layer);
        signature_len = config.simhash["signature_len"],
        sample_ratio = Float(config.simhash["sample_ratio"]),
        input_size = input_dim,
    )

    layer_1 = Dense(input_dim, n_neurons_per_layer[1], relu)
    layer_2 =
        SlideLayer(n_neurons_per_layer[1], n_neurons_per_layer[2], lsh_params[2], identity)
    network = SlideNetwork(layer_1, layer_2)

    learning_rate = 0.0001
    optimizer = AdamOptimizer(eta = learning_rate)

    logger = get_logger(dataset_config["logger"], dataset_config["name"])

    global_logger(logger)

    function ht_update_callback(i, network)
        if i % HT_REINIT_PERIOD == 0
            hashtable_reinit!(network)
        elseif i % HT_UPDATE_PERIOD == 0
            hashtable_update!(network)
        end
    end

    function test_accuracy_callback(i, network)
        if i % test_parameters["test_frequency"] == 0
            test_accuracy(
                network,
                test_set,
                test_parameters["n_test_batches"],
                test_parameters["topk"],
            )
        end
    end

    function checkpoint_callback(i, network)
        if i % save_parameters["save_frequency"] == 0
            Network.save(network, save_parameters["model_path"])
        end
    end

    train!(
        train_loader,
        network,
        optimizer,
        logger;
        n_epochs = 3,
        callbacks = [ht_update_callback, test_accuracy_callback, checkpoint_callback],
        use_zygote = use_zygote,
    )

    println("DONE \n")

    save(logger)
end
