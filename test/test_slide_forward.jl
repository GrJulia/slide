using Test

using Slide
using Slide.Network
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.Hash: LshParams
using Slide.LSH: retrieve

common_lsh = LshParams(n_buckets = 1, n_tables = 1, max_bucket_len = 2)
simparams = LshSimHashParams(common_lsh, 3, 1, 3)

@testset "slide_forward" begin
    x = Array{Float}([1.0; 2.0; 3.0])
    y = Vector{Float}(rand(1:2, 1))
    y_cat = one_hot(y)
    y_cat ./= sum(y_cat, dims = 1)

    n_layers = 1
    neuron_1 = Neuron(
        1,
        Array{Float}([1.0, 1.0, 1.0]),
        zero(Float),
        zeros(Float, 1),
        zeros(Float, 1),
        AdamAttributes(zeros(3), 0, zeros(3), 0),
        false,
    )
    neuron_2 = Neuron(
        2,
        Array{Float}([0.0, 0.0, 0.0]),
        one(Float),
        zeros(Float, 1),
        zeros(Float, 1),
        AdamAttributes(zeros(3), 0, zeros(3), 0),
        false,
    )

    network =
        SlideNetwork([Layer(1, [neuron_1, neuron_2], simparams, identity; batch_size = 1)])

    @views begin
        @test length(network.layers) == 1
        @testset "forward single sample" begin
            forward_single_sample(x[:, 1], network, 1)
            active_neurons = network.layers[1].active_neurons[1]
            @test network.layers[1].output[1] == [6.0, 1.0][active_neurons]
        end
        @test retrieve(network.layers[1].hash_tables.lsh, x[:, 1]) == Set{Int}([1, 2])
    end
end
