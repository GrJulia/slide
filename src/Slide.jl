module Slide

include("common_aliases.jl")

include("flux/Flux.jl")
include("hash/Hash.jl")
include("network/Network.jl")
include("network/slide_zygote/ZygoteNetwork.jl")


end # Slide
