module Logger

export get_logger,
    SlideLogger,
    step!,
    save,
    log_dot_product_metrics,
    precision_at_k,
    compute_avg_dot_product

include("slide_logger.jl")
include("utils.jl")

end
