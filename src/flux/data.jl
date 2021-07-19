using Flux
using Random
using DataLoaders
using LearnBase

struct SparseDataset
    xs::Tuple{Vector{Vector{Int}},Vector{Vector{Float32}}}
    ys::Vector{Vector{Int}}
    n_features::Int
    n_classes::Int
end

function LearnBase.getobs(ds::SparseDataset, idx::Int)
    (xs_indices, xs_vals) = ds.xs
    x = zeros(Float32, ds.n_features)
    x[xs_indices[idx]] = xs_vals[idx]

    ys = ds.ys[idx]
    y = zeros(Int, ds.n_classes)
    y[ys] .= 1

    return x, y
end

LearnBase.nobs(ds::SparseDataset) = length(ds.ys)

function preprocess_dataset(dataset_path)
    f = open(dataset_path, "r")
    x_indices, x_vals, ys = [], [], []
    for line in readlines(f)[2:end]
        line_split = split(line)
        x = map(
            ftr -> (parse(Int, split(ftr, ':')[1]) + 1, parse(Float32, split(ftr, ':')[2])),
            line_split[2:end],
        )
        y = parse.(Int, split(line_split[1], ',')) .+ 1
        push!(x_indices, first.(x))
        push!(x_vals, last.(x))
        push!(ys, y)
        break
    end

    perm = randperm(length(ys))
    data, labels = (x_indices[perm], x_vals[perm]), ys[perm]

    return data, labels
end

function get_dataloaders(config::Dict{String,Any})
    train_data, train_labels = preprocess_dataset(config["dataset"]["train_path"])
    test_data, test_labels = preprocess_dataset(config["dataset"]["test_path"])

    trainset =
        SparseDataset(train_data, train_labels, config["n_features"], config["n_classes"])
    testset =
        SparseDataset(test_data, test_labels, config["n_features"], config["n_classes"])

    train_loader = DataLoader(trainset, config["batch_size"], partial = false)
    test_loader = DataLoader(testset, config["batch_size"], partial = true)
    return train_loader, test_loader
end
