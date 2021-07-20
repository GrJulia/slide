using Base: @kwdef

abstract type Optimizer end

function optimizer_step(optimizer::Optimizer, neuron::OptimizerNeuron)
    error("unimplemented")
end

@kwdef mutable struct AdamOptimizer <: Optimizer
    eta::Float = 0.01
    beta_1::Float = 0.9
    beta_2::Float = 0.999
    epsilon::Float = 1e-8
    t::Int = 1
end

function optimizer_step(optimizer::AdamOptimizer, neuron::OptimizerNeuron)
    dw = sum(neuron.neuron.weight_gradients, dims = 2)[:, 1]
    db = sum(neuron.neuron.bias_gradients)
    adam_attributes = neuron.optimizer_attributes
    adam_attributes.m_dw = optimizer.beta_1 .* adam_attributes.m_dw .+ (1 - optimizer.beta_1) .* dw
    adam_attributes.m_db = optimizer.beta_1 * adam_attributes.m_db + (1 - optimizer.beta_1) * db

    adam_attributes.v_dw = optimizer.beta_2 .* adam_attributes.v_dw .+ (1 - optimizer.beta_2) .* (dw .^ 2)
    adam_attributes.v_db = optimizer.beta_2 * adam_attributes.v_db + (1 - optimizer.beta_2) * (db^2)

    corr_momemntum_dw = adam_attributes.m_dw ./ (1 - optimizer.beta_1^optimizer.t)
    corr_momemntum_db = adam_attributes.m_db / (1 - optimizer.beta_1^optimizer.t)
    corr_velocity_dw = adam_attributes.v_dw ./ (1 - optimizer.beta_2^optimizer.t)
    corr_velocity_db = adam_attributes.v_db / (1 - optimizer.beta_2^optimizer.t)

    neuron.neuron.weight .-=
        optimizer.eta .*
        (corr_momemntum_dw ./ (sqrt.(corr_velocity_dw) .+ optimizer.epsilon))
    neuron.neuron.bias -=
        optimizer.eta * (corr_momemntum_db / (sqrt(corr_velocity_db) + optimizer.epsilon))

    optimizer.t += 1
end
