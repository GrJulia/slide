using Test

using Slide
using Slide.Hash: AbstractLshParams
using Slide.LSH: AbstractHasher, Lsh
using Slide.Network: SlideHashTables, update!

import Slide.Hash.init_lsh!
import Slide.LSH.compute_signatures!

struct MockParams <: AbstractLshParams end
struct SumHasher <: AbstractHasher{SubArray{Float}} end

compute_signatures!(signatures::SubArray{Int}, ::SumHasher, elem::SubArray{Float}) =
    for i = 1:length(signatures)
        signatures[i] = i * sum(elem)
    end

init_lsh!(::MockParams, r, t) = Lsh(3, 10, 10, SumHasher(), SubArray{Float}, Id)

neuron_with_id(i) = Neuron(
    1,
    Array{Float}([1.0, 1.0, 1.0]),
    zero(Float),
    zeros(Float, 1),
    zeros(Float, 1),
    AdamAttributes(zeros(3), 0, zeros(3), 0),
    false,
)


@testset "HashTables" begin
    lsh = init_lsh!(MockParams(), 1, 1)

    tables = SlideHashTables(
        lsh = lsh,
        lsh_params = MockParams(),
        hashes = zeros(Int, 3, 4),
        changed_ids = Set{Id}([1, 3]),
    )

    update!(
        tables,
        [neuron_with_id(1), neuron_with_id(2), neuron_with_id(3), neuron_with_id(4)],
    )

    @test tables.hashes == [
        3 0 3 0
        6 0 6 0
        9 0 9 0
    ]
    @test tables.changed_ids == Set{Id}()
end
