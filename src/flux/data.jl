using Flux
using Random
using DataLoaders
using LearnBase
using DataSets

using Slide


struct SparseDataset
    xs::Tuple{Vector{Vector{Int}},Vector{Vector{Float}}}
    ys::Vector{Vector{Int}}
    batch_size::Int
    n_features::Int
    n_classes::Int
    keep_last::Bool
end

LearnBase.getobs!(
    buffer::Tuple{Matrix{Float},Matrix{Float}},
    ds::SparseDataset,
    raw_batch_idx,
) = LearnBase.getobs!(
    buffer::Tuple{Matrix{Float},Matrix{Float}},
    ds,
    convert(Int, raw_batch_idx),
)

function LearnBase.getobs!(
    buffer::Tuple{Matrix{Float},Matrix{Float}},
    ds::SparseDataset,
    batch_idx::Int,
)
    batch = ds.batch_size
    if ds.keep_last
        batch = min(batch, length(ds.ys) - batch * (batch_idx - 1))
    end

    xs_indices, xs_vals = ds.xs
    x, y = buffer
    x .= zero(Float)
    y .= zero(Float)

    for b = 1:batch
        idx = (batch_idx - 1) * batch + b
        x[xs_indices[idx], b] = xs_vals[idx]
        y[ds.ys[idx], b] .= one(Float)
    end
    y ./= sum(y, dims = 1)
    return buffer
end

LearnBase.getobs(ds::SparseDataset, raw_batch_idx) =
    LearnBase.getobs(ds, convert(Int, raw_batch_idx))

function LearnBase.getobs(ds::SparseDataset, batch_idx::Int)
    batch = ds.batch_size
    if ds.keep_last
        batch = min(batch, length(ds.ys) - batch * (batch_idx - 1))
    end

    xs_indices, xs_vals = ds.xs
    x = zeros(Float, ds.n_features, batch)
    y = zeros(Float, ds.n_classes, batch)

    for b = 1:batch
        idx = (batch_idx - 1) * batch + b
        x[xs_indices[idx], b] = xs_vals[idx]
        y[ds.ys[idx], b] .= one(Float)
    end
    y ./= sum(y, dims = 1)
    return x, y
end

function LearnBase.nobs(ds::SparseDataset)
    n_of_batches = length(ds.ys) / ds.batch_size
    round_fn = if ds.keep_last
        ceil
    else
        floor
    end
    convert(Int, round_fn(n_of_batches))
end

function preprocess_dataset(dataset_path, shuffle)
    f = open(String, dataset(dataset_path))
    lines = split(f, '\n')
    x_indices, x_vals, ys = [], [], []
    for line in lines[2:end-1]
    # f = open(dataset_path, "r")
    # x_indices, x_vals, ys = [], [], []
    # for line in readlines(f)[2:end]
        line_split = split(line)
        x = map(
            ftr -> (parse(Int, split(ftr, ':')[1]) + 1, parse(Float, split(ftr, ':')[2])),
            line_split[2:end],
        )
        y = parse.(Int, split(line_split[1], ',')) .+ 1
        push!(x_indices, first.(x))
        push!(x_vals, last.(x))
        push!(ys, y)
    end

    perm = if shuffle
        randperm(length(ys))
    else
        1:length(ys)
    end
    data, labels = (x_indices[perm], x_vals[perm]), ys[perm]

    return data, labels
end

function get_dataloaders(config::Dict{String,Any})
    train_data, train_labels = preprocess_dataset(config["dataset"]["train_path"], true)
    test_data, test_labels =
        preprocess_dataset(config["dataset"]["test_path"], config["dataset"]["shuffle"])

    train_set = SparseDataset(
        train_data,
        train_labels,
        config["batch_size"],
        config["n_features"],
        config["n_classes"],
        true,
    )
    test_set = SparseDataset(
        test_data,
        test_labels,
        config["batch_size"],
        config["n_features"],
        config["n_classes"],
        false,
    )

    train_loader = eachobsparallel(train_set)
    return train_loader, test_set
end
