using Base: @kwdef
using Statistics: mean
using LinearAlgebra.BLAS: axpy!

using Slide: Float, Id, FloatVector


struct AdamAttributes <: AbstractOptimizerAttributes
    m_dw::Matrix{Float}
    m_db::Vector{Float}
    v_dw::Matrix{Float}
    v_db::Vector{Float}
    n_weight_updates::Vector{Int}
end

AdamAttributes(input_dim::Int, output_dim::Int) = AdamAttributes(
    zeros(input_dim, output_dim),
    zeros(output_dim),
    zeros(input_dim, output_dim),
    zeros(output_dim),
    zeros(Int, output_dim),
)

@kwdef struct AdamOptimizer <: AbstractOptimizer
    eta::Float = 0.01
    beta_1::Float = 0.9
    beta_2::Float = 0.999
    epsilon::Float = 1e-8
end

function optimizer_step!(
    optimizer::AdamOptimizer,
    adam_attributes::AdamAttributes,
    neuron_id::Id,
    weight::T,
    bias::Ref{Float},
    weight_gradients::U,
    bias_gradients::P,
) where {T<:FloatVector,P<:FloatVector,U<:FloatVector}
    dw = weight_gradients
    db = mean(bias_gradients)

    adam_attributes.n_weight_updates[neuron_id] += 1
    t = adam_attributes.n_weight_updates[neuron_id]

    @views begin
        @. adam_attributes.m_dw[:, neuron_id] =
            optimizer.beta_1 * adam_attributes.m_dw[:, neuron_id] +
            (1 - optimizer.beta_1) * dw
        adam_attributes.m_db[neuron_id] =
            optimizer.beta_1 * adam_attributes.m_db[neuron_id] + (1 - optimizer.beta_1) * db

        @. adam_attributes.v_dw[:, neuron_id] =
            optimizer.beta_2 * adam_attributes.v_dw[:, neuron_id] +
            (1 - optimizer.beta_2) * (dw^2)
        adam_attributes.v_db[neuron_id] =
            optimizer.beta_2 * adam_attributes.v_db[neuron_id] +
            (1 - optimizer.beta_2) * (db^2)

        corr_momentum_db = adam_attributes.m_db[neuron_id] / (1 - optimizer.beta_1^t)
        corr_velocity_db = adam_attributes.v_db[neuron_id] / (1 - optimizer.beta_2^t)

        @. weight -=
            optimizer.eta * (
                (adam_attributes.m_dw[:, neuron_id] / (1 - optimizer.beta_1^t)) / (
                    sqrt(adam_attributes.v_dw[:, neuron_id] / (1 - optimizer.beta_2^t)) +
                    optimizer.epsilon
                )
            )
        bias[] -=
            optimizer.eta *
            (corr_momentum_db / (sqrt(corr_velocity_db) + optimizer.epsilon))
    end
end
