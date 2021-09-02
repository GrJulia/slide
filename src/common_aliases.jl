
export Float, Id, LshBatch, FloatVector

const Float = haskey(ENV, "USE_FLOAT64") ? Float64 : Float32
const Id = Int
const SubVector{T} = SubArray{T,1}
const FloatVector = AbstractVector{Float}
const LshBatch = Vector{Tuple{<:FloatVector,Id}}
