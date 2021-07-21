using Test
using Slide.SimHash: SimHasher, signature, initialize!


sh(x, y) = SimHasher(convert(Matrix{Int8}, x), y)

@testset "Signature tests" begin
    @testset "No throw tests" begin
        @test signature(
            sh(hcat([1, 1, 1], [-1, -1, -1]), hcat([1, 2, 3], [1, 2, 3])),
            [1, 1, 1],
        ) == [1, 0]
        @test signature(
            sh(hcat([1, -1, 1], [1, -1, -1]), hcat([1, 2, 3], [1, 3, 2])),
            [1, 2, 3],
        ) == [1, 0]
        @test signature(
            sh(hcat([1, -1, -1, 1], [1, 1, 1, -1]), hcat([1, 4, 3, 2], [1, 2, 3, 4])),
            [1, 1, 1, 4],
        ) == [0, 0]
        @test signature(sh(hcat([-1, 1], [-1, 1]), hcat([1, 4], [1, 2])), [4, 3, 1, 5]) ==
              [1, 0]
    end

    @testset "Throw tests" begin
        @test_throws AssertionError signature(
            sh(hcat([1, 1, 1], [-1, -1, -1]), hcat([1, 2, 3], [1, 2, 3])),
            [1],
        )
        @test_throws AssertionError signature(
            sh(hcat([1, 1, 1], [-1, -1, -1]), hcat([1, 2, 3], [1, 2, 3])),
            [1, 4],
        )
        @test_throws AssertionError signature(sh(hcat([1, 1]), hcat([1, 1])), [1])
    end
end

@testset "SimHasher construction test" begin
    @testset "Throws on dimensions mismatch" begin
        @test_throws ErrorException sh(hcat([1, -1, 1]), hcat([1, 2, 3, 4]))
        @test_throws ErrorException sh(hcat([1, 1], [1, 1]), hcat([1, 2]))
    end

    @testset "Initialize" begin
        using Random: default_rng
        rng = default_rng()

        function test_routine(r, n_hashes, sample_size, data_size)
            sh = initialize!(r, n_hashes, sample_size, data_size)
            size(sh.hashes) == (sample_size, n_hashes) &&
                size(sh.samples) == (sample_size, n_hashes)
        end

        @testset "No throw" begin
            @test test_routine(rng, 10, 12, 14)
            @test test_routine(rng, 11, 7, 14)
            @test test_routine(rng, 1000, 1232, 1400)
        end

        @testset "Throw" begin
            @test_throws AssertionError test_routine(rng, 10, 15, 14)
            @test_throws AssertionError test_routine(rng, 11, 7, 5)
            @test_throws AssertionError test_routine(rng, 1000, 1232, 1100)
        end
    end
end
