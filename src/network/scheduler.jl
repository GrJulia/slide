
abstract type AbstractScheduler end

struct VanillaScheduler <: AbstractScheduler
    update_period::Int
end

function (s::VanillaScheduler)(update, it::Int)
    if it % s.update_period == 0
        update()
    end
end
