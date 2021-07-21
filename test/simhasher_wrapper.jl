using Test
using Random: default_rng

using Slide.Hash: LshParams, init_lsh!
using Slide.LshSimHashWrapp: LshSimHashParams
using Slide.LSH: add!, retrieve


@testset "Initializing LSH with SimHasher" begin
    lsh_params = LshParams(n_buckets = 10, max_bucket_len = 10, n_tables = 100)
    simhash_params = LshSimHashParams(lsh_params, 10, 4, 3)

    lsh_with_simhash = init_lsh!(simhash_params, default_rng(), Int)

    @testset "SimHasher is initialize properly for number of tables" begin
        @test size(lsh_with_simhash.hash.hasher.hashes) == (3, 100 * 4)
        @test size(lsh_with_simhash.hash.hasher.samples) == (3, 100 * 4)
    end

    @testset "LSH can handle vectors" begin
        @test begin
            add!(lsh_with_simhash, zeros(Float32, 10), 10)

            retrieve(lsh_with_simhash, zeros(Float32, 10)) == Set{Int}(10)
        end
    end

    @testset "Throw with bad arguments" begin
        simhash_params = LshSimHashParams(lsh_params, 10, 65, 3)
        @test_throws ErrorException init_lsh!(simhash_params, default_rng(), Int)
    end
end
