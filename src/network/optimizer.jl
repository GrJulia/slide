using Base: @kwdef

abstract type Optimizer end

function optimizer_step(optimizer::Optimizer, neuron::Neuron)
    error("unimplemented")
end

@kwdef mutable struct AdamOptimizer <: Optimizer
    eta::Float = 0.01
    beta_1::Float = 0.9
    beta_2::Float = 0.999
    epsilon::Float = 1e-8
    t::Int = 1
end

function optimizer_step(optimizer::AdamOptimizer, neuron::Neuron)
    dw = mean(neuron.weight_gradients, dims = 2)[:, 1]
    db = mean(neuron.bias_gradients)
    neuron.m_dw = optimizer.beta_1 .* neuron.m_dw .+ (1 - optimizer.beta_1) .* dw
    neuron.m_db = optimizer.beta_1 * neuron.m_db + (1 - optimizer.beta_1) * db

    neuron.v_dw = optimizer.beta_2 .* neuron.v_dw .+ (1 - optimizer.beta_2) .* (dw .^ 2)
    neuron.v_db = optimizer.beta_2 * neuron.v_db + (1 - optimizer.beta_2) * (db ^ 2)

    corr_momemntum_dw = neuron.m_dw ./ (1 - optimizer.beta_1 ^ optimizer.t)
    corr_momemntum_db = neuron.m_db / (1 - optimizer.beta_1 ^ optimizer.t)
    corr_velocity_dw = neuron.v_dw ./ (1 - optimizer.beta_2 ^ optimizer.t)
    corr_velocity_db = neuron.v_db / (1 - optimizer.beta_2 ^ optimizer.t)

    neuron.weight .-= optimizer.eta .* (corr_momemntum_dw ./ (sqrt.(corr_velocity_dw) .+ optimizer.epsilon))
    neuron.bias -= optimizer.eta * (corr_momemntum_db / (sqrt(corr_velocity_db) + optimizer.epsilon))

    optimizer.t += 1
end
