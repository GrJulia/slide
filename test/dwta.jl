using Test
using Slide.DWTA: DWTAHasher, Signatures, initialize!, signatures, EMPTY_SAMPLING, two_universal_hash, MAX_N_ATTEMPS
using Random: default_rng, randperm


@testset "DWTAHasher - computing signatures" begin
    n_tables, n_bins, k, data_len = 1, 6, 3, 2
    idxs_to_list_of_bins = [1, 4, 5, 1, 3, 6, 2, 5, 3, 6, 2, 6, 3, 5, 1, 4, 2, 4]
    n_bins_per_idx_offsets = [0, 3, 6, 8, 10, 12, 13, 14, 16, 18]

    dwta = DWTAHasher(
        idxs_to_list_of_bins,
        n_bins_per_idx_offsets,
        n_tables * n_bins,
        1,
        n_bins,
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
    for (n_tables, n_bins) in [(20, 2), (30, 4), (50, 6)]
        n_hashes = n_tables * n_bins
        log_n_hashes = ceil(Int32, log2(n_hashes))
        dwta = DWTAHasher([], [], n_hashes, log_n_hashes, n_bins)

        hashes = zeros(n_hashes)
        n_attempts = min(n_hashes, MAX_N_ATTEMPS)
        good = true
        for bin_idx = 1:n_hashes
            bin_hashes = zeros(n_hashes)
            for cnt = 1:n_attempts
                hash = two_universal_hash(dwta, Int32(bin_idx), Int32(cnt))
                hashes[hash] += 1
                bin_hashes[hash] += 1
            end
            if any(map(n_hits -> n_hits > 2 * ceil(n_attempts / n_hashes), bin_hashes))
                good = false
            end
        end
        # println("L=$n_tables, K=$n_bins, hashes=$hashes")
        res = res && all(map(n_hits -> n_hits > n_attempts * 0.8, hashes)) && good
    end
    @test res
end

@testset "DWTAHasher - initialization" begin
    rng = default_rng()

    function test_routine(rng, n_tables, n_bins, k, data_len)
        dwta = initialize!(rng, n_tables * n_bins, n_bins, k, data_len)
        n_indices = n_tables * n_bins * k
        length(dwta.idxs_to_list_of_bins) == n_indices && dwta.n_bins_per_idx_offsets[end] == n_indices
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
