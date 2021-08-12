using Test

using Random: seed!

using Slide
using Slide.Network
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.Hash: LshParams

seed!(0)

@testset "slide_backward" begin
    epsilon, threshold = if Float == Float64
        1e-5, 1e-8
    else
        Float32(1e-1), Float32(1e-4)
    end

    n_layers = 2
    n_buckets = 10
    batch_size = 1
    input_dim = 16
    output_dim = 16

    common_lsh = LshParams(n_buckets = n_buckets, n_tables = 10, max_bucket_len = 128)
    lsh_params = [
        LshSimHashParams(common_lsh, input_dim, 2, input_dim ÷ 2),
        LshSimHashParams(common_lsh, 32, 2, 16),
    ]

    network_params = Dict(
        "n_layers" => n_layers,
        "n_neurons_per_layer" => [32, output_dim],
        "layer_activations" => ["relu", "identity"],
        "input_dim" => input_dim,
        "lsh_params" => lsh_params,
    )

    x = rand(Float, input_dim, batch_size)
    y = Vector{Float}(rand(1:output_dim, batch_size))
    network = build_network(network_params, batch_size)
    y_cat = one_hot(y, output_dim)
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
                    epsilon,
                ) < threshold
            end
        end
    end
    for layer in network.layers
        for neuron in layer.neurons
            @test numerical_gradient_bias(network, layer.id, neuron.id, x, y_cat, epsilon) <
                  threshold
        end
    end
end
