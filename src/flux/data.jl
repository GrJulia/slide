using Flux
using Random
using DataLoaders
using LearnBase
using DataSets
using SparseArrays: sparsevec, sparse

using Slide: Float, SparseArray


function parse_line(line)
    line_split = split(line)
    x = map(
        ftr -> (parse(Int, split(ftr, ':')[1]) + 1, parse(Float, split(ftr, ':')[2])),
        line_split[2:end],
    )
    x_indices, x_vals = first.(x), last.(x)
    y = parse.(Int, split(line_split[1], ',')) .+ 1
    x_indices, x_vals, y
end

function preprocess_dataset(
    dataset_path,
    n_features,
    n_classes,
    batch_size,
    shuffle,
)::Vector{Tuple{SparseArray,SparseArray}}
    f = open(String, dataset(dataset_path))
    lines = split(f, '\n')[2:end-1]

    n_samples = length(lines)
    perm = if shuffle
        randperm(n_samples)
    else
        1:n_samples
    end
    lines = lines[perm]

    batches = []
    batched_lines = Iterators.partition(lines, batch_size)
    for raw_batch in batched_lines
        curr_batch_size = length(raw_batch)
        x_Is, x_Js, x_Vs = [], [], []
        y_Is, y_Js, y_Vs = [], [], []
        for (i, raw_line) in enumerate(raw_batch)
            x_indices, x_vals, y = parse_line(raw_line)

            x_Is = vcat(x_Is, x_indices)
            x_Js = vcat(x_Js, fill(i, length(x_indices)))
            x_Vs = vcat(x_Vs, x_vals)

            y_Is = vcat(y_Is, y)
            y_Js = vcat(y_Js, fill(i, length(y)))
            y_Vs = vcat(y_Vs, ones(Float, length(y)))
        end
        x_sparse = sparse(x_Is, x_Js, x_Vs, n_features, curr_batch_size)
        y_sparse = sparse(y_Is, y_Js, y_Vs, n_classes, curr_batch_size)
        push!(batches, (x_sparse, y_sparse))
    end

    return batches
end

function get_dataloaders(config::Dict{String,Any})
    train_set = preprocess_dataset(
        config["dataset"]["train_path"],
        config["n_features"],
        config["n_classes"],
        config["batch_size"],
        true,
    )
    test_set = preprocess_dataset(
        config["dataset"]["test_path"],
        config["n_features"],
        config["n_classes"],
        config["batch_size"],
        true,
    )

    train_loader = DataLoader(train_set, nothing)

    println("Data loaded!")

    return train_loader, test_set
end
