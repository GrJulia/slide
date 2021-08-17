
export Float, Id, LshBatch, FloatVector

const Float = haskey(ENV, "USE_FLOAT64") ? Float64 : Float32
const Id = Int
const LshBatch = Vector{Tuple{SubArray{Float},Id}}
const FloatVector = AbstractVector{Float}
