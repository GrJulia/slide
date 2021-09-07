using Flux
using Random
using DataLoaders
import LearnBase
using DataSets

using Slide


struct SparseDataset
    xs::Tuple{Vector{Vector{Int}},Vector{Vector{Float}}}
    ys::Vector{Vector{Int}}
    batch_size::Int
    n_features::Int
    n_classes::Int
    keep_last::Bool
    smooth_labels::Bool
end

LearnBase.getobs!(
    buffer::Tuple{Vector{Float},Vector{Float}},
    ds::SparseDataset,
    raw_batch_idx,
) = LearnBase.getobs!(
    buffer,
    ds,
    convert(Int, raw_batch_idx),
)

function LearnBase.getobs!(
    buffer::Tuple{Vector{Float},Vector{Float}},
    ds::SparseDataset,
    batch_idx::Int,
)
    xs_indices, xs_vals = ds.xs
    x, y = buffer
    x .= zero(Float)
    y .= zero(Float)

    x[xs_indices[batch_idx]] .= xs_vals[batch_idx]
    y[ds.ys[batch_idx]] .= one(Float)

    if ds.smooth_labels
        y ./= sum(y)
    end

    buffer
end

LearnBase.getobs(ds::SparseDataset, raw_batch_idx) =
    LearnBase.getobs(ds, convert(Int, raw_batch_idx))

function LearnBase.getobs(ds::SparseDataset, batch_idx::Int)
    x = zeros(Float, ds.n_features)
    y = zeros(Float, ds.n_classes)

    LearnBase.getobs!((x, y), ds, batch_idx)
end

LearnBase.nobs(ds::SparseDataset) = length(ds.ys)

function preprocess_dataset(dataset_path, shuffle)
    f = open(dataset_path, "r")
    x_indices, x_vals, ys = [], [], []
    for line in readlines(f)[2:end-1]
        line_split = split(line)
        x = map(
            ftr ->
                (parse(Int, split(ftr, ':')[1]) + 1, parse(Float, split(ftr, ':')[2])),
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
        config["smooth_labels"],
    )
    test_set = SparseDataset(
        test_data,
        test_labels,
        config["batch_size"],
        config["n_features"],
        config["n_classes"],
        false,
        config["smooth_labels"],
    )

    train_loader = DataLoader(train_set, config["batch_size"], buffered=true)
    test_loader = DataLoader(test_set, config["batch_size"], buffered=true)

    train_loader, test_loader
end
