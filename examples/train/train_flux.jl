using Flux
using Flux.Optimise: update!
using Flux.Losses: logitcrossentropy
using JSON
using Statistics
using Random
using NNlib
using CUDA
using Logging: global_logger

using Slide: Float, SparseFloatArray
using Slide.DataLoading
using Slide.Logger: get_logger, step!

"""
Usage:
julia --project=. -t <n_of_threads> examples/train/train_flux.jl examples/configs/<config_name>
"""

function accuracy(out::Matrix{Float}, sparse_labels::SparseFloatArray, top_k::Int)
    labels = Matrix(sparse_labels)
    batch_size, acc = size(out, 2), zero(Float)
    for b = 1:batch_size
        top_k_classes = partialsortperm((@view out[:, b]), 1:top_k, rev = true)
        acc += sum(labels[top_k_classes, b]) / top_k
    end
    return acc / batch_size
end

function train_step!(model, params, opt, device, x::SparseFloatArray, y::SparseFloatArray)
    loss = zero(Float)
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
        convert(Int, length(train_loader)), zero(Float), time_ns(), Flux.params(model)
    for (it, (x, y)) in enumerate(train_loader)
        step!(logger)

        train_stats = if device == gpu
            @timed CUDA.@sync train_step!(model, params, opt, device, x, y)
        else
            @timed train_step!(model, params, opt, device, x, y)
        end

        t1 = time_ns()
        @info "train_step + data loading time" (t1 - t0) / 1.0e9

        loss = train_stats[1]
        total_loss += loss

        @info "train_step time" train_stats[2]
        @info "train_loss" loss log_to_tb = true

        if it % config["testing"]["test_freq"] == 0
            test_acc = test_epoch(model, test_set, config["testing"])
            println("Iteration $it/$n_iters, test_acc=$test_acc")
        end

        t0 = time_ns()
    end
end

function test_epoch(model, test_set, config)
    n_batches = min(config["n_batches"], length(test_set))
    if config["use_random_indices"]
        test_indices = rand(1:length(test_set), n_batches)
    else
        test_indices = 1:n_batches
    end

    total_loss, acc = zero(Float), zero(Float)
    for (x, y) in test_set[test_indices]
        x = device(x)
        out = model(x) |> cpu
        loss = logitcrossentropy(out, y)
        acc += accuracy(out, y, config["top_k_classes"])
        total_loss += loss
    end

    test_loss = total_loss / n_batches
    test_acc = acc / n_batches

    @info "test_loss" test_loss log_to_tb = true
    @info "test_acc" test_acc log_to_tb = true

    return test_acc
end


config = JSON.parsefile(ARGS[1])
config["name"] *= "_flux_" * randstring(8)
println("Name: $(config["name"])")

device = if config["use_gpu"]
    gpu
else
    cpu
end

logger = get_logger(config["logger"], config["name"])
global_logger(logger)

train_loader, test_set = get_dataloaders(config)

model =
    Chain(
        Dense(config["n_features"], config["hidden_dim"], relu, init = Flux.glorot_normal),
        Dense(config["hidden_dim"], config["n_classes"], init = Flux.glorot_normal),
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
