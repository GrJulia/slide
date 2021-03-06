using Test

@testset "SimHash" begin
    include("simhash.jl")
end

@testset "DWTA" begin
    include("dwta.jl")
end

@testset "Network" begin
    include("test_slide_forward.jl")
    include("test_slide_backward.jl")
end

@testset "LSH" begin
    include("lsh.jl")
end

@testset "Lsh-Simhash" begin
    include("simhasher_wrapper.jl")
end

@testset "Lsh-DWTA" begin
    include("dwta_wrapper.jl")
end
