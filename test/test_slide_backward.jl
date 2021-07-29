using Test

using Slide.Network

@testset "slide_backward" begin
    n_layers = 2
    n_buckets = 2
    batch_size = 128
    input_dim = 16
    output_dim = 16
    hash_tables = [HashTable([[] for _ = 1:n_buckets]) for _ = 1:n_layers]
    network_params = Dict(
        "n_layers" => n_layers,
        "n_neurons_per_layer" => [32, output_dim],
        "layer_activations" => ["relu", "identity"],
        "input_dim" => input_dim,
        "hash_tables" => hash_tables,
    )

    x = rand(Float, input_dim, batch_size)
    y = Vector{Float}(rand(1:output_dim, batch_size))
    network = build_network(network_params, batch_size)
    y_cat = one_hot(y)
    y_cat ./= sum(y_cat, dims = 1)

    for layer in network.layers
        for neuron in layer.neurons
            for weight_index = 1:length(neuron.weight)
                @test numerical_gradient_weights(
                    network,
                    layer.id,
                    neuron.id,
                    weight_index,
                    x,
                    y_cat,
                    0.00001,
                ) < 1e-8
            end
        end
    end

    for layer in network.layers
        for neuron in layer.neurons
            @test numerical_gradient_bias(network, layer.id, neuron.id, x, y_cat, 0.00001) < 1e-8
        end
    end
end
