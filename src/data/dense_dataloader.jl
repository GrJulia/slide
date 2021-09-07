using Random
using DataLoaders
using LearnBase
using DataSets

using Slide: Float
using Slide.DataLoading: read_dataset


struct DenseDataset
    xs::Tuple{Vector{Vector{Int}},Vector{Vector{Float}}}
    ys::Vector{Vector{Int}}
    batch_size::Int
    n_features::Int
    n_classes::Int
    keep_last::Bool
    smooth_labels::Bool
end

LearnBase.getobs(ds::DenseDataset, raw_batch_idx) =
    LearnBase.getobs(ds, convert(Int, raw_batch_idx))

function LearnBase.getobs(ds::DenseDataset, batch_idx::Int)
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

    if ds.smooth_labels
        y ./= sum(y, dims = 1)
    end

    return x, y
end

function LearnBase.nobs(ds::DenseDataset)
    n_of_batches = length(ds.ys) / ds.batch_size
    round_fn = if ds.keep_last
        ceil
    else
        floor
    end
    convert(Int, round_fn(n_of_batches))
end

function get_dense_dataloaders(config::Dict{String,Any})
    train_data, train_labels = read_dataset(config["dataset"]["train_path"])
    test_data, test_labels = read_dataset(config["dataset"]["test_path"])

    train_set = DenseDataset(
        train_data,
        train_labels,
        config["batch_size"],
        config["n_features"],
        config["n_classes"],
        true,
        config["smooth_labels"],
    )
    test_set = DenseDataset(
        test_data,
        test_labels,
        config["batch_size"],
        config["n_features"],
        config["n_classes"],
        false,
        config["smooth_labels"],
    )

    train_loader = DataLoader(train_set, nothing)
    train_loader, test_set
end
