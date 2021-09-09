using Random
using DataLoaders
using LearnBase
using DataSets
using SparseArrays: sparsevec, sparse

using Slide: Float, SparseFloatArray


function parse_line(line)
    line_split = split(line)

    x = map(line_split[2:end]) do ftr
        idx, val = split(ftr, ':')
        parse(Int, idx) + 1, parse(Float, val)
    end

    x_indices, x_vals = first.(x), last.(x)
    y = parse.(Int, split(line_split[1], ',')) .+ 1

    x_indices, x_vals, y
end

function read_dataset(dataset_path)
    f = open(String, dataset(dataset_path))
    lines = split(f, '\n')[2:end-1]

    perm = randperm(length(lines))

    map(parse_line, lines[perm])
end

function preprocess_dataset(
    dataset_path,
    n_features,
    n_classes,
    batch_size,
    smooth_labels,
)::Vector{Tuple{SparseFloatArray,SparseFloatArray}}
    lines = read_dataset(dataset_path)
    batched_lines = Iterators.partition(lines, batch_size)

    batches = []
    for raw_batch in batched_lines
        curr_batch_size = length(raw_batch)
        x_row_indices, x_col_indices, x_values = [], [], []
        y_row_indices, y_col_indices, y_values = [], [], []
        for (i, line) in enumerate(raw_batch)
            x_indices, x_vals, y = line

            x_row_indices = vcat(x_row_indices, x_indices)
            x_col_indices = vcat(x_col_indices, fill(i, length(x_indices)))
            x_values = vcat(x_values, x_vals)

            y_row_indices = vcat(y_row_indices, y)
            y_col_indices = vcat(y_col_indices, fill(i, length(y)))

            y_vals = ones(Float, length(y))
            if smooth_labels
                y_vals ./= length(y)
            end
            y_values = vcat(y_values, y_vals)
        end
        x_sparse =
            sparse(x_row_indices, x_col_indices, x_values, n_features, curr_batch_size)
        y_sparse =
            sparse(y_row_indices, y_col_indices, y_values, n_classes, curr_batch_size)
        push!(batches, (x_sparse, y_sparse))
    end

    batches
end

function get_sparse_datasets(config::Dict{String,Any})
    train_set = preprocess_dataset(
        config["dataset"]["train_path"],
        config["n_features"],
        config["n_classes"],
        config["batch_size"],
        config["smooth_labels"],
    )
    test_set = preprocess_dataset(
        config["dataset"]["test_path"],
        config["n_features"],
        config["n_classes"],
        config["batch_size"],
        config["smooth_labels"],
    )

    train_set, test_set
end
