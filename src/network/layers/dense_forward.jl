using SparseArrays: sparsevec

using Slide: Float, Id


function (dense::Dense)(x::A; args...) where {A<:AbstractMatrix{Float}}
    dense.output .= dense.activation.(dense.weights'*x .+ dense.biases)

    dense.output
end

function (dense::Dense)(x::Vector{SlideOutput}; args...)
    batch_size = length(x)
    input_size = size(dense.weights, 1)

    sparse_input = spzeros(Float, Id, input_size, batch_size)

    for (i, (vals, ids)) in enumerate(x)
        sparse_input[:, i] = sparsevec(ids, vals, input_size)
    end

    dense(sparse_input)
end
