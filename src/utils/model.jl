type Model
    f::Formula
    dict::Dict{Symbol, Any}
end

function Base.show(io::IO, m::Model)
    println(io, m.f)
    for (k, v) in m.dict
        println(io, k, ": ", v)
    end
end

macro model(args...)
    Expr(:call, :model_helper, (esc(Base.Meta.quot(a)) for a in args)...)
end

function model_helper(args...)
    (args[1].head === :call && args[1].args[1] === :(~)) || throw("First argument of @model should be a formula")
    f = Formula(args[1].args[2], args[1].args[3])
    dict = Dict{Symbol, Any}()
    for i in 2:length(args)
        isa(args[i], Expr) &&  args[i].head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
        dict[args[i].args[1]] = args[i].args[2]
    end
    Model(f, dict)
end


