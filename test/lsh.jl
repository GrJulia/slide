using Slide.LSH: AbstractHasher, Lsh, add!, retrieve
import Slide.LSH

struct MockHasher <: AbstractHasher{Int}
    query_signature::Function
    signature::Function
end

LSH.signature(m::MockHasher, elem::Int)::Vector{Int} = m.signature(elem)
LSH.query_signature(m::MockHasher, elem::Int)::Vector{Int} = m.query_signature(elem)


@testset "Adding to the LSH" begin
    signature(elem) = [(elem + i - 1) % 5 + 1 for i = 1:5]
    lsh() = Lsh(5, 4, 5, MockHasher(signature, signature), Int)

    function test_table(table, bucket_id, expected)
        bucket = table.buckets[bucket_id]
        bucket == expected
    end

    @testset "Simple add" begin
        l = lsh()
        add!(l, 10)

        @test test_table(l.hash_tables[1], 1, [10])
        @test test_table(l.hash_tables[2], 2, [10])
        @test test_table(l.hash_tables[3], 3, [10])
        @test test_table(l.hash_tables[4], 4, [10])
        @test test_table(l.hash_tables[5], 1, [10])

        add!(l, 15)

        @test test_table(l.hash_tables[1], 1, [10, 15])
        @test test_table(l.hash_tables[2], 2, [10, 15])
        @test test_table(l.hash_tables[3], 3, [10, 15])
        @test test_table(l.hash_tables[4], 4, [10, 15])
        @test test_table(l.hash_tables[5], 1, [10, 15])

        add!(l, 13)

        @test test_table(l.hash_tables[1], 1, [10, 15])
        @test test_table(l.hash_tables[2], 2, [10, 15])
        @test test_table(l.hash_tables[3], 3, [10, 15])
        @test test_table(l.hash_tables[4], 4, [10, 15])
        @test test_table(l.hash_tables[5], 1, [10, 15])

        @test test_table(l.hash_tables[1], 4, [13])
        @test test_table(l.hash_tables[2], 1, [13])
        @test test_table(l.hash_tables[3], 1, [13])
        @test test_table(l.hash_tables[4], 2, [13])
        @test test_table(l.hash_tables[5], 3, [13])
    end

    @testset "Adding more than bucket_size to bucket" begin
        l = lsh()

        add!(l, 10)
        add!(l, 15)
        add!(l, 20)
        add!(l, 25)
        add!(l, 30)

        @test test_table(l.hash_tables[1], 1, [10, 15, 20, 25, 30])

        add!(l, 35)

        @test test_table(l.hash_tables[1], 1, [15, 20, 25, 30, 35])
    end
end

@testset "Retrieving elements from lsh" begin
    signature(elem) = [(elem + i - 1) % 5 + 1 for i = 1:5]
    lsh() = Lsh(5, 5, 5, MockHasher(signature, signature), Int)

    l = lsh()
    add!(l, 10)

    @test retrieve(l, 1) == Set{Int}()
    @test retrieve(l, 2) == Set{Int}()
    @test retrieve(l, 3) == Set{Int}()
    @test retrieve(l, 4) == Set{Int}()
    @test retrieve(l, 5) == Set(10)

    add!(l, 15)
    add!(l, 11)

    @test retrieve(l, 1) == Set(11)
    @test retrieve(l, 2) == Set{Int}()
    @test retrieve(l, 3) == Set{Int}()
    @test retrieve(l, 4) == Set{Int}()
    @test retrieve(l, 5) == Set([10, 15])

    add!(l, 14)
    add!(l, 16)

    @test retrieve(l, 1) == Set([11, 16])
    @test retrieve(l, 2) == Set{Int}()
    @test retrieve(l, 3) == Set{Int}()
    @test retrieve(l, 4) == Set(14)
    @test retrieve(l, 5) == Set([10, 15])
end
