using Test

@testset "SimHash" begin
    include("simhash.jl")
end

@testset "Network" begin
    include("test_slide_forward.jl")
end
