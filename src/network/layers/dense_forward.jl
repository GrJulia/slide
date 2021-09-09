using Slide: Id, Float, FloatVector
using LinearAlgebra: dot


function forward_single_sample!(
    layer::Dense{F,O},
    input::U,
    x_index::Int,
    ::Any,
) where {F,O,U<:FloatVector}
    layer.output[:, x_index] = layer.activation.(layer.weights' * input + layer.biases)

    @view layer.output[:, x_index]
end

function forward_single_sample!(
    layer::Dense{F,O},
    input::Tuple{U,P},
    x_index::Int,
    ::Any,
) where {F,O,U<:FloatVector,P<:AbstractVector{Id}}
    sparse_input, activated_neuron_ids = input

    @views layer.output[:, x_index] =
        layer.activation.(
            layer.weights[activated_neuron_ids, :]' * sparse_input + layer.biases,
        )

    @view layer.output[:, x_index]
end
