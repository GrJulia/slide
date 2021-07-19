using TensorBoardLogger
using Logging
using DataStructures

mutable struct Logger
    curr_it::Int
    logs::DefaultDict
    tb_logger::TBLogger
    incr_tb::Bool

    Logger(tb_logger) = new(0, DefaultDict([]), tb_logger, false)
end

function step(logger::Logger)
    logger.curr_it += 1
    logger.incr_tb = false
    if logger.curr_it % 20 == 0
        save(logger)
    end
end

function log_scalar(logger::Logger, key::String, val::Any, log_to_tb=false)
    push!(logger.logs[key], (logger.curr_it, val))
    if log_to_tb
        if !logger.incr_tb
            @info key key = val
            logger.incr_tb = true
        else
            @info key key = val log_step_increment=0
        end
    end
end

function save(logger::Logger)

end

function get_logger(config)
    tb_logger = TBLogger(joinpath(config["logging_path"], config["name"]))
    global_logger(tb_logger)

    logger = Logger(tb_logger)
    return logger
end