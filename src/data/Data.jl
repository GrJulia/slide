module DataLoading

export read_dataset, get_sparse_dataloaders, get_dense_dataloaders

include("sparse_dataloader.jl")
include("dense_dataloader.jl")

end
