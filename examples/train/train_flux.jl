using Flux
using Flux.Optimise: update!
using Flux.Losses: logitcrossentropy
using JSON
using Statistics
using Random
using LearnBase
using NNlib
using CUDA

using Slide.FluxTraining

"""
Usage:
julia --project=. -t <n_of_threads> examples/train_flux.jl examples/configs/<config_name>
"""

function accuracy(out::Matrix{Float32}, labels::Matrix{Float32}, top_k::Int)
    batch_size, acc = size(out, 2), zero(Float32)
    for b = 1:batch_size
        top_k_classes = partialsortperm((@view out[:, b]), 1:top_k, rev = true)
        acc += sum(labels[top_k_classes, b]) / top_k
    end
    return acc / batch_size
end

function train_step!(model, params, opt, device, x::Matrix{Float32}, y::Matrix{Float32})
    loss = zero(Float32)
    x = device(x)
    y = device(y)
    grads = gradient(params) do
        out = model(x)
        loss = logitcrossentropy(out, y)
        loss
    end
    update!(opt, params, grads)
    return loss
end

function train_epoch!(model, train_loader, test_set, opt, device, config, logger)
    n_iters, total_loss, t0, params =
        convert(Int, length(train_loader)), zero(Float32), time_ns(), Flux.params(model)
    for (it, (x, y)) in enumerate(train_loader)
        FluxTraining.step!(logger)

        train_stats = @timed CUDA.@sync train_step!(model, params, opt, device, x, y)

        t1 = time_ns()
        log_scalar!(logger, "train_step + data loading time", (t1 - t0) / 1.0e9)

        loss = train_stats[1]
        total_loss += loss

        log_scalar!(logger, "train_step time", train_stats[2])
        log_scalar!(logger, "train_loss", loss, true)

        if it % config["testing"]["test_freq"] == 0
            test_acc = test_epoch(model, test_set, logger, config["testing"])
            @info "Iteration $it/$n_iters, test_acc=$test_acc"
        end

        t0 = time_ns()
    end
end

function test_epoch(model, test_set, logger, config)
    n_batches = min(config["n_batches"], LearnBase.nobs(test_set))
    if config["use_random_indices"]
        rand_indices = rand(1:LearnBase.nobs(test_set), n_batches)
    else
        rand_indices = 1:n_batches
    end

    total_loss, acc = zero(Float32), zero(Float32)
    for idx in rand_indices
        x, y = LearnBase.getobs(test_set, idx)
        x = device(x)
        out = model(x) |> cpu
        loss = logitcrossentropy(out, y)
        acc += accuracy(out, y, config["top_k_classes"])
        total_loss += loss
    end

    test_loss = total_loss / n_batches
    test_acc = acc / n_batches

    log_scalar!(logger, "test_loss", test_loss, true)
    log_scalar!(logger, "test_acc", test_acc, true)

    return test_acc
end


config = JSON.parsefile(ARGS[1])
config["name"] *= "_" * randstring(8)
println("Name: $(config["name"])")

device = if config["use_gpu"]
    gpu
else
    cpu
end

logger = get_logger(config)
train_loader, test_set = get_dataloaders(config)

model = Chain(
    Dense(config["n_features"], config["hidden_dim"], relu, init=Flux.glorot_normal),
    Dense(config["hidden_dim"], config["n_classes"], init=Flux.glorot_normal),
) |> device

opt = ADAM(config["lr"])

Flux.@epochs config["n_epochs"] train_epoch!(
    model,
    train_loader,
    test_set,
    opt,
    device,
    config,
    logger,
)
