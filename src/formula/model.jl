struct Model
    f::FormulaTerm
    dict::Dict{Symbol, Any}
end

function Base.show(io::IO, m::Model)
    println(io, m.f)
    for (k, v) in m.dict
        println(io, k, ": ", v)
    end
end

"""
Capture and parse a set of expressions to generate a Model
## Arguments
* `ex::Expr`: an expression parsed as a `Formula`
* `fe::Expr`: Fixed effect expression.   You can add an arbitrary number of high dimensional fixed effects, separated with `+`.  Interact multiple categorical variables using `&` .     Interact a categorical variable with a continuous variable using `&`.   Alternative, use `*` to add a categorical variable and its interaction with a continuous variable. Variables must be of type CategoricalArray (use `categorical` to convert a variable to a `CategoricalArray`).
* `vcov::Expr`: Vcov formula. Default to `simple`. `robust` and `cluster()` are also implemented
* `weights::Expr`: Weight variable. Corresponds to analytical weights
* `subset::Expr`: Expression of the form State .>= 30

### Returns
* `::Model`: a `Model` struct

## Detail
A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, and instruments
```
depvar ~ exogeneousvars + (endogeneousvars ~ instrumentvars
```

### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df.StateC =  categorical(df.State)
df.YearC =  categorical(df.Year)
reg(df, @model(Sales ~ NDI, weights = Pop))
@model(Sales ~ NDI, fe = StateC, vcov = robust)
@model(Sales ~ NDI, fe = StateC + YearC, weights = Pop, vcov = cluster(StateC))

```
"""
macro model(ex, kws...)
    f = @eval(@formula($ex))
    dict = Dict{Symbol, Any}()
    for kw in kws
       isa(kw, Expr) &&  kw.head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
       dict[kw.args[1]] = kw.args[2]
    end
    Model(f, dict)
end




