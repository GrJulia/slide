
export Float, Id, LshBatch, FloatVector

const Float = haskey(ENV, "USE_FLOAT64") ? Float64 : Float32
const Id = Int
const SubVector{T} = SubArray{T, 1}
const LshBatch = Vector{Tuple{SubVector{Float},Id}}
const FloatVector = AbstractVector{Float}
