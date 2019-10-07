struct ModelTerm
    f::FormulaTerm
    dict::Dict{Symbol, Any}
end

ModelTerm(f::FormulaTerm; kwargs...) = ModelTerm(f, Dict(pairs(kwargs)...))
function Base.show(io::IO, m::ModelTerm)
    println(io, m.f)
    for (k, v) in m.dict
        println(io, k, ": ", v)
    end
end

import StatsModels: capture_call
macro model(ex, kws...)
    f = StatsModels.terms!(StatsModels.sort_terms!(StatsModels.parse!(ex)))
    d = Dict{Symbol, Any}()
    for kw in kws
       isa(kw, Expr) &&  kw.head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
       if kw.args[1] == :fe
            @warn "The keyword argument fe is deprecated. Instead of @model(y ~ x, fe = state + year),  write @formula(y ~ x + fe(state) + fe(year))"
            d[:feformula] = kw.args[2]
        elseif kw.args[1] == :ife
                 @warn "The keyword argument ife is deprecated. Instead of @model(y ~ x, ife = (state + year, 2)),  write @formula(y ~ x + ife(state, year, 2))"
            d[:ifeformula] = kw.args[2]
        elseif kw.args[1] == :vcov
            d[:vcovformula] = kw.args[2] 
            @warn "The keyword argument vcov is deprecated. Instead of reg(df, @model(y ~ x, vcov = cluster(State))),  write reg(df, @formula(y ~ x), Vcov.cluster(:State))"
        elseif kw.args[1] == :subset
            d[:subsetformula] = kw.args[2] 
            @warn "The keyword argument subset is deprecated. Instead of reg(df, @model(y ~ x, subset = State .>= 30),  write reg(df, @formula(y ~ x), subset = df.State .>= 30))"
        elseif kw.args[1] == :weight
                 d[:weight] = kw.args[2] 
                 @warn "The keyword argument weight is deprecated. Instead of reg(df, @model(y ~ x, weight = Pop),  write reg(df, @formula(y ~ x), weight = :Pop)"
        else
           d[kw.args[1]] = kw.args[2]
       end
    end
    :(ModelTerm($f, $d))
end


function evaluate_subset(df, ex::Expr)
    if ex.head == :call
        return Expr(ex.head, ex.args[1], (evaluate_subset(df, ex.args[i]) for i in 2:length(ex.args))...)
    else
        return Expr(ex.head, (evaluate_subset(df, ex.args[i]) for i in 1:length(ex.args))...)
    end
end
evaluate_subset(df, ex::Symbol) = df[!, ex]
evaluate_subset(df, ex)  = ex


