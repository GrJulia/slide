module Optimizers

export AbstractOptimizerAttributes, AbstractOptimizer, AdamOptimizer, optimizer_step!, optimizer_end_epoch_step!

abstract type AbstractOptimizer end
abstract type AbstractOptimizerAttributes end

# TODP update
function optimizer_step!(optimizer::AbstractOptimizer)
    error("unimplemented")
end

function optimizer_end_epoch_step!(optimizer::AbstractOptimizer)
    error("unimplemented")
end

include("adam.jl")

end # Optimizers
