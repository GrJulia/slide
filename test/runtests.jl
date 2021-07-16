using Test

@testset "Dummy test" begin
    @test 1 == 1
end

@testset "Network" begin
    include("test_slide_forward.jl")
end
