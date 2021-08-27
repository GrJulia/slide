module SlideLogger

export get_logger, Logger, step!, save, log_dot_product_metrics, precision_at_k

include("logger.jl")
include("utils.jl")

end
