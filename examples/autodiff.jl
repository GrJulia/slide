using ForwardDiff


function f(x)
    return sum(x)
end

function g(x)
    ForwardDiff.gradient(f, x)
end

x = [1.0 2.0 3.0]
println(f(x))
println(g(x))