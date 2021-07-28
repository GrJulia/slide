sigmoid(x::Vector{Float})::Vector{Float} = 1 ./ (1 .+ exp.(.-x))

identity(x::Any)::Any = x

relu(x::Vector{Float})::Vector{Float} = max.(0, x)

function negative_sparse_logit_cross_entropy(
    output::Array{Float},
    y_true::Array{Float},
    output_activated_neuron_ids::Vector,
)
    logit_cross_entropy = Vector{Float}()
    saved_softmax = Vector{Vector{Float}}()
    for i = 1:size(output)[end]
        current_loss, current_softmax = sparse_logit_cross_entropy_sample(
            (@view output[:, i][output_activated_neuron_ids[i]]),
            (@view y_true[:, i][output_activated_neuron_ids[i]]),
        )
        push!(logit_cross_entropy, current_loss)
        push!(saved_softmax, current_softmax) # warning: saved_softmax only contains the softmax values
        # for activated neurons
    end
    return -mean(logit_cross_entropy), saved_softmax
end

function sparse_logit_cross_entropy_sample(
    output::A,
    y_true::A,
) where {A<:AbstractArray{Float}}
    λ, argmax_output = findmax(output)
    sparse_exp_output = map(a -> exp(a - λ), output)
    # use of log1p after reading: https://github.com/mitmath/18335/blob/spring19/psets/midtermsol.pdf
    return sum(
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
    return x * (1 - x)
end

@inline function gradient(
    ::Type{typeof(negative_sparse_logit_cross_entropy)},
    labels::Float,
    probabilities::Float,
    ratio_positve_labels_sampled::Float
)
    return probabilities * ratio_positve_labels_sampled - labels
end

@inline function gradient(::Type{typeof(relu)}, x::Float)
    return Int(x > 0)
end
