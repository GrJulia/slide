using Flux
using Flux.Optimise: update!
using Flux.Losses: crossentropy
using JSON
using Statistics
using TensorBoardLogger
using Logging
using BSON: @save

using Slide.FluxTraining

"""
Usage:
julia --threads <n_of_threads> src/flux/train.jl flux_config.json
"""

MA_WEIGHT = 0.9

ma(total, curr) = MA_WEIGHT * total + (1 - MA_WEIGHT) * curr

function accuracy(out, labels, top_k)
    batch_size, acc = size(out, 2), 0
    for b = 1:batch_size
        top_k_classes = partialsortperm((@view out[:, b]), 1:top_k, rev = true)
        acc += sum(labels[top_k_classes, b]) / top_k
    end
    return acc / batch_size
end

function train_epoch(model, train_loader, opt, config)
    params = Flux.params(model)
    n_iters, losses, total_acc = length(train_loader), nothing, nothing
    for (it, (x, y)) in enumerate(train_loader)
        if it % 10 == 0
            println("Iteration $it/$n_iters, loss=", losses)
        end
        loss, out = nothing, nothing
        grads = gradient(params) do
            out = model(x)
            loss = crossentropy(out, y)
            loss
        end
        losses = isnothing(losses) ? loss : ma(losses, loss)
        acc = accuracy(out, y, config["top_k_classes"])
        total_acc = isnothing(total_acc) ? acc : ma(total_acc, acc)
        @info "train_loss" train_loss = losses
        @info "train_acc" train_acc = total_acc log_step_increment = 0
        update!(opt, params, grads)
    end
end

function test_epoch(model, test_loader, config)
    losses, acc = 0, 0
    for (it, (x, y)) in enumerate(test_loader)
        out = model(x)
        loss = crossentropy(out, y)
        acc += accuracy(out, y, config["top_k_classes"])
        losses += loss
    end

    @info "test_loss" test_loss = losses / length(test_loader) log_step_increment = 0
    @info "test_acc" test_acc = acc / length(test_loader) log_step_increment = 0
    @save "$(joinpath(config["logging_path"], config["name"], "last_checkpoint.bson"))" model
end


config = JSON.parsefile(ARGS[1])

tb_logger = TBLogger(joinpath(config["logging_path"], config["name"]))
global_logger(tb_logger)

train_loader, test_loader = get_dataloaders(config)

model = Chain(
    Dense(config["n_features"], config["hidden_dim"]),
    Dense(config["hidden_dim"], config["n_classes"]),
    softmax,
)

opt = ADAM(config["lr"])

n_epochs = config["n_epochs"]
for ep = 1:n_epochs
    println("\nEpoch $ep")
    train_epoch(model, train_loader, opt, config)
    test_epoch(model, test_loader, config)
end