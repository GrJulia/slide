using Test
using Random: default_rng

using Slide: Float
using Slide.Hash: LshParams, init_lsh!
using Slide.LshAsymmetricHasher: LshAsymHasherParams
using Slide.LshSimHashWrapper: LshSimHashParams
using Slide.LSH: add!, add_batch!, retrieve

# TODO: better tests (including tests checking if ALSH approximately solves MIPS)
@testset "Initializing ALSH with SimHasher" begin
    lsh_params = LshParams(n_buckets = 10, max_bucket_len = 10, n_tables = 100)
    simhash_params = LshSimHashParams(lsh_params, 10, 4, 3)
    asym_hasher_params = LshAsymHasherParams(simhash_params, 2)

    alsh_with_simhash = init_lsh!(asym_hasher_params, default_rng(), Int)

    @test @views begin
        r = rand(Float, 10)[:]
        add!(alsh_with_simhash, r, 10)

        retrieve(alsh_with_simhash, r) == Set{Int}([10])
    end
end
