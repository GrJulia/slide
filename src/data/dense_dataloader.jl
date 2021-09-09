using DataLoaders
using LearnBase

using Slide: Float, SparseFloatArray
using Slide.DataLoading: get_sparse_datasets


struct DenseDataset
    xs::Vector{SparseFloatArray}
    ys::Vector{SparseFloatArray}
    DenseDataset(dataset) = new(first.(dataset), last.(dataset))
end

LearnBase.getobs(ds::DenseDataset, raw_batch_idx) =
    LearnBase.getobs(ds, convert(Int, raw_batch_idx))

LearnBase.getobs(ds::DenseDataset, batch_idx::Int) =
    Matrix(ds.xs[batch_idx]), Matrix(ds.ys[batch_idx])

LearnBase.nobs(ds::DenseDataset) = length(ds.xs)

function get_dense_dataloaders(config::Dict{String,Any})
    sparse_train_set, sparse_test_set = get_sparse_datasets(config)
    train_set = DenseDataset(sparse_train_set)
    test_set = DenseDataset(sparse_test_set)

    train_loader = DataLoader(train_set, nothing)
    train_loader, test_set
end
