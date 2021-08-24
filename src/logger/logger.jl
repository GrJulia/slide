using TensorBoardLogger
using Logging
using DataStructures
using JSON


mutable struct Logger <: AbstractLogger
    curr_it::Int
    logs::DefaultDict
    tb_logger::TBLogger
    log_path::String
    incr_tb::Bool

    Logger(tb_logger, log_dir) =
        new(0, DefaultDict(() -> []), tb_logger, joinpath(log_dir, "logs.json"), false)
end

function step!(logger::Logger)
    logger.curr_it += 1
    logger.incr_tb = false
    if logger.curr_it % 20 == 0
        save(logger)
    end
end

function log_scalar!(logger::Logger, key::String, val::Any, log_to_tb = false)
    push!(logger.logs[key], (logger.curr_it, val))
    if log_to_tb
        with_logger(logger.tb_logger) do
            if !logger.incr_tb
                @info key key = val
                logger.incr_tb = true
            else
                @info key key = val log_step_increment = 0
            end
        end
    end
end

function save(logger::Logger)
    logs_string = JSON.json(logger.logs)
    open(logger.log_path, "w") do f
        JSON.print(f, logs_string, 4)
    end
end

function get_logger(config)
    log_dir = config["logging_path"] * "/" * config["name"]
    tb_logger = TBLogger(log_dir)

    logger = Logger(tb_logger, log_dir)
    return logger
end
