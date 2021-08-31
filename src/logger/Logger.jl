module SlideLogger

export get_logger, Logger, step!, save, log_dot_product_metrics, precision_at_k, compute_avg_dot_product

include("logger.jl")
include("utils.jl")

end
