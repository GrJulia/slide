using Test
using Slide.DWTA: DWTAHasher, Signatures, initialize!
using Random: default_rng


@testset "DWTAHasher initialization test" begin
    @testset "Initialize" begin
        rng = default_rng()

        function test_routine(rng, n_tables, n_bins, k, data_len)
            dwta = initialize!(rng, n_tables, n_bins, k, data_len)
            n_indices = n_tables * n_bins * k
            length(dwta.idx_to_bins) == n_indices && dwta.bins_offsets[end] == n_indices
        end

        @testset "Dense indices" begin
            @test test_routine(rng, 50, 8, 6, 30)
            @test test_routine(rng, 100, 10, 15, 100)
        end

        @testset "Sparse indices" begin
            @test test_routine(rng, 2, 3, 2, 100)
            @test test_routine(rng, 5, 4, 10, 10000)
        end
    end
end
