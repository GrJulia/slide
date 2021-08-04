
abstract type AbstractScheduler end

"""
    PeriodicScheduler schedules update every `update_period` iterations.
"""
struct PeriodicScheduler <: AbstractScheduler
    update_period::Int
end

function (s::PeriodicScheduler)(update, it::Int)
    if it % s.update_period == 0
        update()
    end
end
