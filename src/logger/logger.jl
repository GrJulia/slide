using TensorBoardLogger
import Logging
using DataStructures
using JSON


mutable struct Logger <: Logging.AbstractLogger
    curr_it::Int
    logs::DefaultDict
    tb_logger::TBLogger
    log_path::String
    incr_tb::Bool
    log_train_metrics::Bool

    Logger(tb_logger, log_dir, log_train_metrics) = new(
        0,
        DefaultDict(() -> []),
        tb_logger,
        joinpath(log_dir, "logs.json"),
        false,
        log_train_metrics,
    )
end

Logging.handle_message(
    logger::Logger,
    level::Logging.LogLevel,
    message::Nothing,
    args...;
    kwargs...,
) = return

function Logging.handle_message(
    logger::Logger,
    level::Logging.LogLevel,
    message::String,
    args...;
    kwargs...,
)
    if isempty(kwargs)
        return
    end

    if length(kwargs) > 2
        error("Expected at most 2 keys, received $(length(kwargs))")
    end

    log_to_tb, log_key, log_val = false, message, nothing
    for (key, val) in pairs(kwargs)
        if key == :log_to_tb
            log_to_tb = val
        else
            log_val = val
        end
    end

    log_scalar!(logger, log_key, log_val, log_to_tb)
end

Logging.shouldlog(logger::Logger, args...) = true

Logging.min_enabled_level(logger::Logger) =
    logger.log_train_metrics ? Logging.Debug : Logging.Info

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
        Logging.with_logger(logger.tb_logger) do
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

    logger = Logger(tb_logger, log_dir, config["log_train_metrics"])
    return logger
end
