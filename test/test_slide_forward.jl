using Test

using Slide.Network
using Slide.Network:
    build_activated_neurons_single_sample, forward_single_sample, sparse_softmax



@testset "slide_forward" begin
    x = Array{Float32}([1.0; 2.0; 3.0])
    n_layers = 1
    neuron_1 = Neuron(1, Array{Float64}([1.0, 1.0, 1.0]), 0, zeros(1))
    neuron_2 = Neuron(2, Array{Float64}([0.0, 0.0, 0.0]), 1, zeros(1))
    network =
        SlideNetwork([Layer(1, [neuron_1, neuron_2], HashTable([[1], [2]]), identity)])

    @test length(network.layers) == 1
    @test forward_single_sample(x[:, 1], network, [[1, 2]]) == [6.0; 1.0]
    @test build_activated_neurons_single_sample(x[1, :], network) in [[[1]], [[2]]]

    network.layers[1] =
        Layer(1, [neuron_1, neuron_2], HashTable([[1], [2]]), sparse_softmax)
    @test sum(forward_single_sample(x[:, 1], network, [[1, 2]])) ≈ 1.0
end
