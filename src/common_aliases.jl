
export Float, Id

const Float = haskey(ENV, "USE_FLOAT64") ? Float64 : Float32
const Id = Int
