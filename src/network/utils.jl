using Statistics

function batch_input(
    x::Matrix{Float32},
    y::Matrix{Int},
    batch_size::Int64,
    drop_last::Bool,
)::Vector{Tuple{Matrix{Float32},Matrix{Int}}}
    batches = map(Iterators.partition(axes(x, 2), batch_size)) do columns
        x[:, columns], y[:, columns]
    end
    if drop_last && size(batches[end])[1] < batch_size
        return batches[1:end-1]
    end
    return batches
end

function one_hot(y::Vector, n_labels::Int64 = maximum(y))
    y_categorical = zeros(Int64, n_labels, length(y))
    for (i, label) in enumerate(y)
        y_categorical[label, i] = 1
    end
    y_categorical
end

function cross_entropy(y_pred, y_true)
    -mean(sum(y_true .* log.(y_pred .+ eps()), dims = 1))
end
