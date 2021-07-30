using Test
using Slide.DWTA: DWTAHasher, Signatures, initialize!, signatures, EMPTY_SAMPLING
using Random: default_rng


@testset "DWTAHasher - computing signatures" begin
    @testset "signatures" begin
        @testset "example from the paper" begin
            n_tables, n_bins, k, data_len = 1, 6, 3, 2
            idxs_to_bins = [1, 4, 5, 1, 3, 6, 2, 5, 3, 6, 2, 6, 3, 5, 1, 4, 2, 4]
            bins_offsets = [0, 3, 6, 8, 10, 12, 13, 14, 16, 18]

            dwta = DWTAHasher(idxs_to_bins, bins_offsets, n_tables * n_bins, n_bins, one(Float32), 1)

            data1 = [0, 0, 5, 0, 0, 7, 6, 0, 0]
            data2 = [0, 0, 1, 0, 0, 0, 0, 0, 0]
                
            out1 = Signatures([[EMPTY_SAMPLING, 1, 3, EMPTY_SAMPLING, 3, EMPTY_SAMPLING]])
            out2 = Signatures([[EMPTY_SAMPLING, 1, EMPTY_SAMPLING, EMPTY_SAMPLING, 2, EMPTY_SAMPLING]])

            @test signatures(dwta, data1) == out1
            @test signatures(dwta, data2) == out2
        end

        @testset "testing densification" begin
            
        end
    end
end


@testset "DWTAHasher - initialization" begin
    @testset "initialize!" begin
        rng = default_rng()

        function test_routine(rng, n_tables, n_bins, k, data_len)
            dwta = initialize!(rng, n_tables, n_bins, k, data_len)
            n_indices = n_tables * n_bins * k
            length(dwta.idx_to_bins) == n_indices && dwta.bins_offsets[end] == n_indices
        end

        @testset "dense indices" begin
            @test test_routine(rng, 50, 8, 6, 30)
            @test test_routine(rng, 100, 10, 15, 100)
        end

        @testset "sparse indices" begin
            @test test_routine(rng, 2, 3, 2, 100)
            @test test_routine(rng, 5, 4, 10, 10000)
        end
    end
end
