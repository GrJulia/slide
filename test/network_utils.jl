using Slide
using Slide.Network
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.Hash: LshParams
using Slide.LSH: retrieve

import Slide.Network.get_active_neurons
import Slide.Network.get_active_neuron_ids

@testset "Network utils" begin
    common_lsh = LshParams(n_buckets = 1, n_tables = 1, max_bucket_len = 2)
    simparams = LshSimHashParams(common_lsh, 3, 1, 3)

    neuron_1 = Neuron(
        1,
        Array{Float}([1.0, 1.0, 1.0]),
        0.0,
        [1, 1, 1, 0],
        zeros(1),
        zeros(1),
        zeros(1, 1),
        zeros(1),
        AdamAttributes(zeros(3), 0, zeros(3), 0),
    )
    neuron_2 = Neuron(
        2,
        Array{Float}([0.0, 0.0, 0.0]),
        1.0,
        [1, 0, 1, 0],
        zeros(1),
        zeros(1),
        zeros(1, 1),
        zeros(1),
        AdamAttributes(zeros(3), 0, zeros(3), 0),
    )

    layer = Layer(1, [neuron_1, neuron_2], simparams, identity)
    network = SlideNetwork([layer])

    @testset "Active Neurons" begin
        @test get_active_neurons(layer, 1) == [neuron_1, neuron_2]
        @test get_active_neurons(layer, 2) == [neuron_1]
        @test get_active_neurons(layer, 3) == [neuron_1, neuron_2]
        @test get_active_neurons(layer, 4) == []

        @test get_active_neuron_ids(network, 1) == [[1, 2], [1], [1, 2], []]
    end
end
