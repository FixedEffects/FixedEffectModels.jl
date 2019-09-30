struct ModelTerm
    f::FormulaTerm
    dict::Dict{Symbol, Any}
end

ModelTerm(f::FormulaTerm; kwargs...) = Modelterm(f, Dict(pairs(kwargs)...))
function Base.show(io::IO, m::ModelTerm)
    println(io, m.f)
    for (k, v) in m.dict
        println(io, k, ": ", v)
    end
end

import StatsModels: capture_call

"""
Capture and parse a set of expressions to generate a Model
## Arguments
* `ex::Expr`: an expression parsed as a `Formula`
* `fe::Expr`: Fixed effect expression.   You can add an arbitrary number of high dimensional fixed effects, separated with `+`.  Interact multiple categorical variables using `&` .     Interact a categorical variable with a continuous variable using `&`.   Alternative, use `*` to add a categorical variable and its interaction with a continuous variable. Variables must be of type CategoricalArray (use `categorical` to convert a variable to a `CategoricalArray`).
* `vcov::Expr`: Vcov formula. Default to `simple`. `robust` and `cluster()` are also implemented
* `weights::Expr`: Weight variable. Corresponds to analytical weights
* `subset::Expr`: Expression of the form State .>= 30

### Returns
* `::ModelTerm`: a `ModelTerm`

## Detail
A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, and instruments
```
depvar ~ exogeneousvars + (endogeneousvars ~ instrumentvars) + fe(high dimentional fixed effects)
```

### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df.StateC = categorical(df.State)
df.YearC = categorical(df.Year)
reg(df, @model(Sales ~ NDI))
@model(Sales ~ NDI, vcov = robust)
@model(Sales ~ NDI, vcov = cluster(StateC))

```
"""
macro model(ex, kws...)
    f = StatsModels.terms!(StatsModels.sort_terms!(StatsModels.parse!(ex)))
    d = Dict{Symbol, Any}()
    for kw in kws
       isa(kw, Expr) &&  kw.head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
       if kw.args[1] == :fe
            @warn "The keyword argument fe is deprecated. Instead of @model(y ~ x, fe = state + year),  write @model(y ~ x + fe(state) + fe(year))"
            d[:feformula] = kw.args[2]
        elseif kw.args[1] == :ife
                 @warn "The keyword argument ife is deprecated. Instead of @model(y ~ x, ife = (state + year, 2)),  write @model(y ~ x + ife(state, year, 2))"
            d[:ifeformula] = kw.args[2]
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


