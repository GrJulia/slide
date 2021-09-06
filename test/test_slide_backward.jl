using Test

using Random: seed!

using Slide
using Slide.Network
using Slide.Network.Layers
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.Hash: LshParams

seed!(0)

@testset "slide_backward" begin
    epsilon, threshold = if Float == Float64
        1e-5, 1e-8
    else
        Float32(1e-1), Float32(1e-4)
    end

    n_buckets = 10
    batch_size = 1
    input_dim = 16
    output_dim = 16

    common_lsh = LshParams(n_buckets = n_buckets, n_tables = 10, max_bucket_len = 128)
    lsh_params = [
        LshSimHashParams(common_lsh, input_dim, 2, input_dim ÷ 2),
        LshSimHashParams(common_lsh, 32, 2, 16),
        LshSimHashParams(common_lsh, 32, 2, 16),
        LshSimHashParams(common_lsh, 32, 2, 16),
    ]

    x = rand(Float, input_dim, batch_size)
    y = Vector{Float}(rand(1:output_dim, batch_size))
    network = SlideNetwork(
        Dense(
            16,
            32,
            relu,
        ),
        SlideLayer(
            32,
            32,
            lsh_params[2],
            relu
        ),
        Dense(
            32,
            32,
            relu,
        ),
        SlideLayer(
            32,
            output_dim,
            lsh_params[4],
            identity
        ),
    )

    y_cat = one_hot(y, output_dim)
    y_cat ./= sum(y_cat, dims = 1)

    for (id, layer) in enumerate(network.layers),
        neuron_id = 1:size(layer.weights, 2),
        weight_index = 1:size(layer.weights, 1)

        @test numerical_gradient_weights(
            network,
            id,
            neuron_id,
            weight_index,
            x,
            y_cat,
            epsilon,
        ) < threshold
    end

    for (id, layer) in enumerate(network.layers), neuron_id = 1:size(layer.weights, 2)
        @test numerical_gradient_bias(network, id, neuron_id, x, y_cat, epsilon) <
              threshold
    end
end
