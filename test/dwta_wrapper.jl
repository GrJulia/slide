using Test
using Random: default_rng

using Slide: Float
using Slide.Hash: LshParams, init_lsh!
using Slide.LshDWTAWrapper: LshDWTAParams
using Slide.LSH: compute_signatures, add!, add_batch!, retrieve


@testset "Initializing LSH with DWTAHasher" begin
    lsh_params = LshParams(n_buckets = 10, max_bucket_len = 10, n_tables = 2)
    dwta_params = LshDWTAParams(lsh_params, 6, 8, 10, false)

    lsh = init_lsh!(dwta_params, default_rng(), Int)

    @testset "DWTAHasher is properly initialized" begin
        @test lsh.hash.hasher.n_hashes == (2 * 6)
        @test size(lsh.hash.hasher.indices_in_bin) == (8, 2 * 6)
    end

    @testset "LSH can handle vectors" begin
        @test @views begin
            add!(lsh, zeros(Float64, 10)[:], 10)
            add_batch!(
                lsh,
                convert(Vector{Tuple{SubArray{Float64},Int}}, [(zeros(Float64, 10)[:], 2)]),
            )

            retrieve(lsh, zeros(Float64, 10)[:]) == Set{Int}([10, 2])
        end
        
        @test @views begin
            r = rand(Float64, 10)
            add!(lsh, r[:], 23)

            retrieve(lsh, r[:]) == Set{Int}([23])
            retrieve(lsh, (r .+ 23)[:]) == Set{Int}([23])
            retrieve(lsh, (r .+ 0.4)[:]) == Set{Int}([23])
            retrieve(lsh, (r .- 0.014)[:]) == Set{Int}([23])
        end
    end
end
