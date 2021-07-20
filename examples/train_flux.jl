using Flux
using Flux.Optimise: update!
using Flux.Losses: logitcrossentropy
using JSON
using Statistics
using Random
using BSON: @save
using LearnBase

using Slide.FluxTraining

"""
Usage:
julia --threads <n_of_threads> src/flux/train.jl flux_config.json
"""

MA_WEIGHT = 0

ma(total, curr) = MA_WEIGHT * total + (1 - MA_WEIGHT) * curr

function accuracy(out, labels, top_k)
    batch_size, acc = size(out, 2), 0
    for b = 1:batch_size
        top_k_classes = partialsortperm((@view out[:, b]), 1:top_k, rev = true)
        acc += sum(labels[top_k_classes, b]) / top_k
    end
    return acc / batch_size
end

function train_step(model, x, y)
    params = Flux.params(model)
    loss = nothing
    grads = gradient(params) do
        out = model(x)
        loss = logitcrossentropy(out, y)
        loss
    end
    update!(opt, params, grads)
    return loss
end

function train_epoch(model, train_loader, test_set, opt, config, logger)
    n_iters, losses, t0 = length(train_loader), nothing, time_ns()
    for (it, (x, y)) in enumerate(train_loader)
        FluxTraining.step(logger)

        train_stats = @timed train_step(model, x, y)

        t1 = time_ns()
        log_scalar(logger, "train_step + data loading time", (t1-t0)/1.0e9)

        loss = train_stats[1]
        losses = isnothing(losses) ? loss : ma(losses, loss)

        log_scalar(logger, "train_step time", train_stats[2])
        log_scalar(logger, "train_step gc time", train_stats[4])
        log_scalar(logger, "train_loss", losses, true)

        if it % config["testing"]["test_freq"] == 0
            println("Iteration $it/$n_iters, loss=", losses)
            test_epoch(model, test_set, logger, config["testing"])
        end

        t0 = time_ns()
    end
end

function test_epoch(model, test_set, logger, config)
    n_batches = config["n_batches"]
    if config["use_random_indices"]
        rand_indices = rand(1:LearnBase.nobs(test_set), n_batches)
    else
        rand_indices = 1:n_batches
    end

    losses, acc = 0, 0
    for idx in rand_indices
        x, y = LearnBase.getobs(test_set, idx)
        out = model(x)
        loss = logitcrossentropy(out, y)
        acc += accuracy(out, y, config["top_k_classes"])
        losses += loss
    end

    test_loss = losses / n_batches
    test_acc = acc / n_batches

    log_scalar(logger, "test_loss", test_loss, true)
    log_scalar(logger, "test_acc", test_acc, true)
end


config = JSON.parsefile(ARGS[1])
config["name"] *= "_" * randstring(8)
println("Name: $(config["name"])")

logger = get_logger(config)
train_loader, test_set = get_dataloaders(config)

model = Chain(
    Dense(config["n_features"], config["hidden_dim"]),
    Dense(config["hidden_dim"], config["n_classes"]),
)

opt = ADAM(config["lr"])

n_epochs = config["n_epochs"]
for ep = 1:n_epochs
    println("\nEpoch $ep")
    train_epoch(model, train_loader, test_set, opt, config, logger)

    @save "$(joinpath(config["logging_path"], config["name"], "last_checkpoint.bson"))" model
end
