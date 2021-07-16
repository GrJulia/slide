using PyCall
using Random

keras = pyimport("tensorflow.keras")
np = pyimport("numpy")


@pydef mutable struct SparseDataset <: keras.utils.Sequence
    function __init__(self, dataset_path, batch_size, n_features, n_classes)
        __init__(self) = pybuiltin(:super)(SparseDataset, self)[:__init__]()
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.__preprocess_dataset(dataset_path)
    end

    function __preprocess_dataset(self, dataset_path)
        f = open(dataset_path, "r")
        x_indices, x_vals, ys = [], [], []
        for (cnt, line) in enumerate(readlines(f)[2:end])
            line_split = split(line)
            x = map(ftr -> (parse(Int, split(ftr, ':')[1]) + 1, parse(Float32, split(ftr, ':')[2])), line_split[2:end])
            y = parse.(Int, split(line_split[1], ',')) .+ 1
            push!(x_indices, first.(x))
            push!(x_vals, last.(x))
            push!(ys, y)
            if cnt == 3000
                break
            end
        end
    
        perm = randperm(length(ys))
        data, labels = (x_indices[perm], x_vals[perm]), ys[perm]
    
        self.xs = data
        self.ys = labels
    end

    function __len__(self)
        return Int(floor(length(self.ys) / self.batch_size))
    end

    function __getitem__(self, b_idx)
        (xs_indices, xs_vals) = self.xs

        x = zeros(Float32, self.n_features, self.batch_size)
        y = zeros(Float32, self.n_classes, self.batch_size)
        for idx in 1:self.batch_size
            tr_idx = b_idx * self.batch_size + idx
            x[xs_indices[tr_idx], idx] = xs_vals[tr_idx]
            
            ys = self.ys[tr_idx]
            y[ys, idx] .= 1
        end

        return PyReverseDims(x), PyReverseDims(y)
    end
end
