using Slide.LSH: AbstractHasher, Lsh, add!, retrieve, add_batch!
import Slide.LSH

struct MockHasher <: AbstractHasher{Int}
    query_signature::Function
    signature::Function
end

LSH.compute_signatures(m::MockHasher, elem::Int)::Vector{Int} = m.signature(elem)
LSH.compute_query_signatures(m::MockHasher, elem::Int)::Vector{Int} =
    m.query_signature(elem)


function LSH.compute_signatures!(
    signature::T,
    m::MockHasher,
    elem::Int,
) where {T<:AbstractArray{Int}}
    x = m.signature(elem)
    for (i, s) in enumerate(x)
        signature[i] = s
    end
end

function LSH.compute_query_signatures!(
    signature::T,
    m::MockHasher,
    elem::Int,
) where {T<:AbstractArray{Int}}
    x = m.query_signature(elem)
    for (i, s) in enumerate(x)
        signature[i] = s
    end
end



@testset "Adding to the LSH" begin
    signature(elem) = [(elem + i - 1) % 5 + 1 for i = 1:5]
    lsh() = Lsh(5, 4, 5, MockHasher(signature, signature), Int, Int)

    function test_table(table, bucket_id, expected)
        bucket = table.buckets[bucket_id]
        bucket == expected
    end

    @testset "Simple add" begin
        l = lsh()
        add!(l, 10, 10)

        @test test_table(l.hash_tables[1], 2, [10])
        @test test_table(l.hash_tables[2], 3, [10])
        @test test_table(l.hash_tables[3], 4, [10])
        @test test_table(l.hash_tables[4], 1, [10])
        @test test_table(l.hash_tables[5], 2, [10])

        add!(l, 15, 15)

        @test test_table(l.hash_tables[1], 2, [10, 15])
        @test test_table(l.hash_tables[2], 3, [10, 15])
        @test test_table(l.hash_tables[3], 4, [10, 15])
        @test test_table(l.hash_tables[4], 1, [10, 15])
        @test test_table(l.hash_tables[5], 2, [10, 15])

        add!(l, 13, 13)

        @test test_table(l.hash_tables[1], 2, [10, 15])
        @test test_table(l.hash_tables[2], 3, [10, 15])
        @test test_table(l.hash_tables[3], 4, [10, 15])
        @test test_table(l.hash_tables[4], 1, [10, 15])
        @test test_table(l.hash_tables[5], 2, [10, 15])

        @test test_table(l.hash_tables[1], 1, [13])
        @test test_table(l.hash_tables[2], 2, [13])
        @test test_table(l.hash_tables[3], 2, [13])
        @test test_table(l.hash_tables[4], 3, [13])
        @test test_table(l.hash_tables[5], 4, [13])
    end

    @testset "Adding more than bucket_size to bucket" begin
        l = lsh()

        add!(l, 10, 10)
        add!(l, 15, 15)
        add!(l, 20, 20)
        add!(l, 25, 25)
        add!(l, 30, 30)

        @test test_table(l.hash_tables[1], 2, [10, 15, 20, 25, 30])

        add!(l, 35, 35)

        @test test_table(l.hash_tables[1], 2, [15, 20, 25, 30, 35])
    end

    @testset "Adding in batch equivalent to sequential adding" begin
        l_sequential = lsh()

        add!(l_sequential, 10, 10)
        add!(l_sequential, 15, 15)
        add!(l_sequential, 20, 20)
        add!(l_sequential, 25, 25)
        add!(l_sequential, 30, 30)

        l_batch = lsh()
        add_batch!(l_batch, [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)])

        for (t_sequential, t_batch) in zip(l_sequential.hash_tables, l_batch.hash_tables)
            @test t_sequential.buckets == t_batch.buckets
        end
    end

    @testset "Batch addition returns correct signatures" begin
        l = lsh()
        batch = [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]
        signatures = add_batch!(l, batch)

        @test signatures == [
            1 2 3 4 5
            2 3 4 5 1
            3 4 5 1 2
            4 5 1 2 3
            5 1 2 3 4
        ]
    end
end

@testset "Retrieving elements from lsh" begin
    signature(elem) = [(elem + i - 1) % 5 + 1 for i = 1:5]
    lsh() = Lsh(5, 5, 5, MockHasher(signature, signature), Int, Int)

    l = lsh()
    add!(l, 10, 10)

    @test retrieve(l, 1) == Set{Int}()
    @test retrieve(l, 2) == Set{Int}()
    @test retrieve(l, 3) == Set{Int}()
    @test retrieve(l, 4) == Set{Int}()
    @test retrieve(l, 5) == Set(10)

    add!(l, 15, 15)
    add!(l, 11, 11)

    @test retrieve(l, 1) == Set(11)
    @test retrieve(l, 2) == Set{Int}()
    @test retrieve(l, 3) == Set{Int}()
    @test retrieve(l, 4) == Set{Int}()
    @test retrieve(l, 5) == Set([10, 15])

    add!(l, 14, 14)
    add!(l, 16, 16)

    @test retrieve(l, 1) == Set([11, 16])
    @test retrieve(l, 2) == Set{Int}()
    @test retrieve(l, 3) == Set{Int}()
    @test retrieve(l, 4) == Set(14)
    @test retrieve(l, 5) == Set([10, 15])
end
