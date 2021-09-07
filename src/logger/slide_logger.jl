import Logging

using JSON
using Logging: LogLevel, Info
using TensorBoardLogger: TBLogger
using DataStructures: DefaultDict


mutable struct SlideLogger <: Logging.AbstractLogger
    curr_it::Int
    logs::DefaultDict
    tb_logger::Union{TBLogger,Nothing}
    log_path::String
    incr_tb::Bool
    min_level::LogLevel

    SlideLogger(tb_logger, log_dir, level = Info) = new(
        0,
        DefaultDict(() -> []),
        tb_logger,
        joinpath(log_dir, "logs.json"),
        false,
        level,
    )
end

Logging.handle_message(
    logger::SlideLogger,
    level::LogLevel,
    message::Nothing,
    args...;
    kwargs...,
) = nothing

function Logging.handle_message(
    logger::SlideLogger,
    level::LogLevel,
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

Logging.shouldlog(::SlideLogger, args...) = true

Logging.min_enabled_level(logger::SlideLogger) = logger.min_level

function step!(logger::SlideLogger)
    logger.curr_it += 1
    logger.incr_tb = false
    if logger.curr_it % 20 == 0
        save(logger)
    end
end

function log_scalar!(logger::SlideLogger, key::String, val::Any, log_to_tb = false)
    push!(logger.logs[key], (logger.curr_it, val))

    !log_to_tb && return nothing
    logger.tb_logger == nothing && error("Logging to tb is disabled")

    Logging.with_logger(logger.tb_logger) do
        if !logger.incr_tb
            @info key key = val
            logger.incr_tb = true
        else
            @info key key = val log_step_increment = 0
        end
    end
end

function save(logger::SlideLogger)
    logs_string = JSON.json(logger.logs)
    open(logger.log_path, "w") do f
        JSON.print(f, logs_string, 4)
    end
end

function get_logger(config, name, log_level = Info)
    log_dir = config["logging_path"] * "/" * name
    mkpath(log_dir)

    tb_logger = config["use_tensorboard"] ? TBLogger(log_dir) : nothing

    SlideLogger(tb_logger, log_dir, log_level)
end
