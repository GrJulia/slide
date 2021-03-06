using Test

using Slide
using Slide.Network
using Slide.Network.Layers
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.Hash: LshParams
using Slide.LSH: retrieve

common_lsh = LshParams(n_buckets = 1, n_tables = 1, max_bucket_len = 2)
simparams = LshSimHashParams(common_lsh, 3, 1, 3)

@testset "slide_forward" begin
    input_dim = 3
    x = Array{Float}([1.0; 2.0; 3.0])
    layer = SlideLayer(input_dim, 2, simparams, identity)
    new_batch!(layer, 1)

    @views begin
        @testset "forward single sample" begin
            @test forward_single_sample!(layer, x[:, 1], 1, nothing)[1] ≈
                  forward_single_sample!(
                layer,
                (x[:, 1], collect(1:input_dim)),
                1,
                nothing,
            )[1]
        end
        @test retrieve(layer.hash_tables.lsh, x[:, 1]) == Set{Int}([1, 2])
    end
end
