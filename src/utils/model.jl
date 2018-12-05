struct Model
    f::Formula
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
df[:StateC] =  categorical(df[:State])
df[:YearC] =  categorical(df[:Year])
reg(df, @model(Sales ~ NDI, weights = Pop))
@model(Sales ~ NDI, fe = StateC, vcov = robust)
@model(Sales ~ NDI, fe = StateC + YearC, weights = Pop, vcov = cluster(StateC)

```
"""
macro model(args...)
    Expr(:call, :model_helper, (esc(Base.Meta.quot(a)) for a in args)...)
end

function model_helper(args...)
    (args[1].head === :call && args[1].args[1] === :(~)) || throw("First argument of @model should be a formula")
    f = @eval(@formula($(args[1].args[2]) ~ $(args[1].args[3])))
    dict = Dict{Symbol, Any}()
    for i in 2:length(args)
        isa(args[i], Expr) &&  args[i].head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
        dict[args[i].args[1]] = args[i].args[2]
    end
    Model(f, dict)
end


