using Test

using Slide.Network

@testset "slide_forward" begin
    x = Array{Float}([1.0; 2.0; 3.0])
    n_layers = 1
    neuron_1 = OptimizerNeuron(
        Neuron(
            1,
            Array{Float32}([1.0, 1.0, 1.0]),
            0,
            zeros(1),
            zeros(1),
            zeros(1, 1),
            zeros(1),
        ),
        AdamAttributes(zeros(3), 0, zeros(3), 0),
    )
    neuron_2 = OptimizerNeuron(
        Neuron(
            2,
            Array{Float32}([0.0, 0.0, 0.0]),
            1,
            zeros(1),
            zeros(1),
            zeros(1, 1),
            zeros(1),
        ),
        AdamAttributes(zeros(3), 0, zeros(3), 0),
    )
    network =
        SlideNetwork([Layer(1, [neuron_1, neuron_2], HashTable([[1], [2]]), identity)])

    @test length(network.layers) == 1
    @test forward_single_sample((@view x[:, 1]), network, [[1, 2]], 1) == [6.0; 1.0]
    @test build_activated_neurons_single_sample((@view x[1, :]), network, true) in [[[1]], [[2]]]
end
