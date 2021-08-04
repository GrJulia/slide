using Test
using Slide.DWTA: DWTAHasher, Signature, initialize!, signature, EMPTY_SAMPLING, two_universal_hash
using Random: default_rng, randperm


@testset "DWTAHasher - computing signatures" begin
    @testset "example from the paper #1" begin
        n_hashes, log_n_hashes, data_len = 1, 1, 4
        indices_in_bin = Matrix{Int32}(undef, (3, 1))
        indices = [
            [4, 1, 2],
        ]
        indices_in_bin[:, 1] = indices[1]

        dwta = DWTAHasher(indices_in_bin, n_hashes, log_n_hashes)

        data1 = [10, 12, 9, 23]
        data2 = [8, 9, 1, 12]
        data3 = [9, 2, 6, 1]
        data4 = [3, 5, 1, 7]

        out1 = Signature([1])
        out2 = Signature([1])
        out3 = Signature([2])
        out4 = Signature([1])

        @test signature(dwta, data1, true) == out1
        @test signature(dwta, data2, true) == out2
        @test signature(dwta, data3, true) == out3
        @test signature(dwta, data4, true) == out4
    end

    @testset "example from the paper #2" begin
        n_hashes, log_n_hashes, data_len = 6, 3, 2
        indices_in_bin = Matrix{Int32}(undef, (3, 6))
        indices = [
            [2, 1, 8],
            [5, 3, 9],
            [6, 2, 4],
            [8, 9, 1],
            [1, 7, 3],
            [2, 4, 5],
        ]
        for i = 1:6
            indices_in_bin[:, i] = indices[i] 
        end

        dwta = DWTAHasher(indices_in_bin, n_hashes, log_n_hashes)

        data1 = [0, 0, 5, 0, 0, 7, 6, 0, 0]
        data2 = [0, 0, 1, 0, 0, 0, 0, 0, 0]

        @testset "wta" begin
            out1 = Signature([EMPTY_SAMPLING, 2, 1, EMPTY_SAMPLING, 2, EMPTY_SAMPLING])
            out2 = Signature([EMPTY_SAMPLING, 2, EMPTY_SAMPLING, EMPTY_SAMPLING, 3, EMPTY_SAMPLING])

            @test signature(dwta, data1, false) == out1
            @test signature(dwta, data2, false) == out2
        end

        @testset "dwta" begin
            out1 = Signature([2, 2, 1, 2, 2, 1])
            out2 = Signature([3, 2, 2, 3, 3, 2])

            @test signature(dwta, data1, true) == out1
            @test signature(dwta, data2, true) == out2
        end
    end
end

@testset "DWTAHasher - 2-universal hashing" begin
    res = true
    for (n_tables, n_bins) in [(20, 2), (30, 4), (50, 6)]
        n_hashes = n_tables * n_bins
        log_n_hashes = ceil(UInt32, log2(n_hashes))
        dwta = DWTAHasher(Matrix{Int32}(undef, (3, 6)), n_hashes, log_n_hashes)

        hashes = zeros(n_hashes)
        n_attempts = min(n_hashes, 100)
        good = true

        for bin_idx = 1:n_hashes
            bin_hashes = zeros(n_hashes)
            for cnt = 1:n_attempts
                hash = two_universal_hash(dwta, UInt32(bin_idx), UInt32(cnt))
                hashes[hash] += 1
                bin_hashes[hash] += 1
            end
            
            if any(map(n_hits -> n_hits > 2 * ceil(n_attempts / n_hashes), bin_hashes))
                good = false
            end
        end

        res = res && all(map(n_hits -> n_hits > n_attempts * 0.8, hashes)) && good
    end
    @test res
end

@testset "DWTAHasher - initialization" begin
    rng = default_rng()

    function test_routine(rng, n_tables, n_bins, k, data_len)
        dwta = initialize!(rng, UInt32(n_tables * n_bins), UInt32(k), UInt32(data_len))
        size(dwta.indices_in_bin) == (k, n_tables * n_bins)
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
