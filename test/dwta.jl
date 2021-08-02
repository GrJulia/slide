using Test
using Slide.DWTA: DWTAHasher, Signatures, initialize!, signatures, EMPTY_SAMPLING, two_universal_hash
using Random: default_rng, randperm
using IterTools


@testset "DWTAHasher - computing signatures" begin
    n_tables, n_bins, k, data_len = 1, 6, 3, 2
    idxs_to_bins = [1, 4, 5, 1, 3, 6, 2, 5, 3, 6, 2, 6, 3, 5, 1, 4, 2, 4]
    bins_per_idx_offsets = [0, 3, 6, 8, 10, 12, 13, 14, 16, 18]

    next_idxs = Matrix{UInt16}(undef, n_bins, n_bins)
    for i = 1:n_bins
        next_idxs[:, i] = randperm(n_bins)
    end

    dwta = DWTAHasher(
        idxs_to_bins,
        bins_per_idx_offsets,
        n_tables * n_bins,
        n_bins,
        next_idxs,
    )

    data1 = [0, 0, 5, 0, 0, 7, 6, 0, 0]
    data2 = [0, 0, 1, 0, 0, 0, 0, 0, 0]

    @testset "example from the paper" begin
        out1 = Signatures([EMPTY_SAMPLING, 1, 3, EMPTY_SAMPLING, 3, EMPTY_SAMPLING])
        out2 = Signatures([EMPTY_SAMPLING, 1, EMPTY_SAMPLING, EMPTY_SAMPLING, 2, EMPTY_SAMPLING])

        @test signatures(dwta, data1, true) == out1
        @test signatures(dwta, data2, true) == out2
    end

    @testset "densification" begin
        # println(signatures(dwta, data1, false))
        # println(signatures(dwta, data2, false)) 
    end
end

@testset "DWTAHasher - 2-universal hashing" begin
    res = true
    for n_bins = 2:15
        next_idxs = Matrix{UInt16}(undef, n_bins, n_bins)
        for i = 1:n_bins
            next_idxs[:, i] = randperm(n_bins)
        end

        dwta = DWTAHasher([], [], 0, n_bins, next_idxs)

        threshold = 0.9
        hashes = zeros(n_bins)
        for (bin_idx, cnt) in product(1:n_bins, 1:n_bins)
            hash = two_universal_hash(dwta, UInt16(bin_idx), UInt16(cnt))
            hashes[hash] += 1
        end
        
        res = res && all(map(n_hits -> n_hits > n_bins * threshold, hashes))
    end
    @test res
end

@testset "DWTAHasher - initialization" begin
    rng = default_rng()

    function test_routine(rng, n_tables, n_bins, k, data_len)
        dwta = initialize!(rng, n_tables * n_bins, n_bins, k, data_len)
        n_indices = n_tables * n_bins * k
        length(dwta.idx_to_bins) == n_indices && dwta.n_bins_per_idx_offsets[end] == n_indices
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
