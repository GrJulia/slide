using TensorBoardLogger
using Logging

mutable struct Logger
    curr_it::Int
    logs::DefaultDict{Vector{Tuple{Int, Any}}}
    tb_logger::TBLogger
    incr_tb::Bool

    Logger(tb_logger) = new(0, DefaultDict([]), tb_logger, false)

    function step()
        curr_it += 1
        incr_tb_logger = false
        if curr_it % 20 == 0
            save()
        end
    end

    function log_scalar(key, val, log_to_tb=false)
        push!(logs[key], (curr_it, val))
        if log_to_tensorboard
            if !incr_tb
                @info key key = val
                incr_tb = true
            else
                @info key key = val log_step_increment=0
            end
    end

    function save()

    end
end

function get_logger(config)
    tb_logger = TBLogger(joinpath(config["logging_path"], config["name"]))
    global_logger(tb_logger)

    logger = Logger(tb_logger)
    return logger
end