import JLD2

struct DenseLayerSerialization{F}
    weights::Matrix{Float}
    bias::Vector{Float}
    activation::F
end

struct SlideLayerSerialization{A,F}
    base::DenseLayerSerialization{F}
    lsh_params::A
end


JLD2.writeas(::Type{Dense}) = DenseLayerSerialization
JLD2.writeas(::Type{SlideLayer}) = SlideLayerSerialization

JLD2.wconvert(::Type{SlideLayerSerialization}, a::SlideLayer) = SlideLayerSerialization(
    DenseLayerSerialization(a.weights, a.bias, a.activation),
    a.hash_tables.lsh_params,
)

JLD2.rconvert(::Type{SlideLayer}, a::SlideLayerSerialization) =
    SlideLayer(a.base.weights, a.base.bias, a.base.activation, a.lsh_params)


JLD2.wconvert(::Type{DenseLayerSerialization}, a::Dense) =
    DenseLayerSerialization(a.weights, a.bias, a.activation)

JLD2.rconvert(::Type{Dense}, a::DenseLayerSerialization) =
    DenseLayer(a.base.weights, a.base.bias, a.base.activation)
