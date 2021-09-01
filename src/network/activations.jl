using Slide: FloatVector

@inline sigmoid(x::Vector{Float})::Vector{Float} = 1 ./ (1 .+ exp.(.-x))

@inline relu(x::T where {T<:FloatVector})::Vector{Float} = max.(0, x)

@inline relu(x::Float)::Float = max(0, x)

function negative_sparse_logit_cross_entropy(
    output::Vector{<:FloatVector},
    y_true::Vector{<:FloatVector},
)::Tuple{Float,Vector{Vector{Float}}}
    output_size = size(output)[end]
    logit_cross_entropy = Vector{Float}(undef, output_size)

    # warning: saved_softmax only contains the softmax values for activated neurons
    saved_softmax = Vector{Vector{Float}}(undef, output_size)

    for i = 1:output_size
        current_loss, current_softmax =
            sparse_logit_cross_entropy_sample(output[i], y_true[i])
        logit_cross_entropy[i] = current_loss
        saved_softmax[i] = current_softmax
    end

    -mean(logit_cross_entropy), saved_softmax
end

function sparse_logit_cross_entropy_sample(
    output::T,
    y_true::U,
)::Tuple{Float,T} where {T<:FloatVector,U<:FloatVector}
    λ, argmax_output = findmax(output)
    sparse_exp_output = map(a -> exp(a - λ), output)

    # use of log1p after reading: https://github.com/mitmath/18335/blob/spring19/psets/midtermsol.pdf
    sum(
        y_true .* (
            (output .- λ) .- log1p(
                sum(
                    i == argmax_output ? 0.0 : sparse_exp_output[i] for
                    i = 1:length(sparse_exp_output)
                ),
            )
        ),
    ),
    sparse_exp_output / sum(sparse_exp_output)
end

activation_name_to_function =
    Dict("identity" => identity, "sigmoid" => sigmoid, "relu" => relu)

@inline function gradient(::Type{typeof(sigmoid)}, x::Float)
    x * (one(Float) - x)
end

@inline function gradient(
    ::Type{typeof(negative_sparse_logit_cross_entropy)},
    labels::Float,
    probabilities::Float,
    ratio_positve_labels_sampled::Float,
)
    probabilities * ratio_positve_labels_sampled - labels
end

@inline function gradient(
    ::Type{typeof(negative_sparse_logit_cross_entropy)},
    labels::T,
    probabilities::U,
    ratio_positve_labels_sampled::Float,
) where {T<:FloatVector,U<:FloatVector}
    @. probabilities * ratio_positve_labels_sampled - labels
end

@inline function gradient(::Type{typeof(relu)}, x::T) where {T<:FloatVector}
    map(z -> Float(z > 0), x)
end

@inline function gradient(::Type{typeof(identity)}, x::T) where {T<:FloatVector}
    ones(Float, length(x))
end
