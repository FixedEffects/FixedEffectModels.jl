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
* `ex`: an expression parsed as a Formula struct
* `fe` : Fixed effect expression.   You can add an arbitrary number of high dimensional fixed effects, separated with `+`.  Interact multiple categorical variables using `&` .     Interact a categorical variable with a continuous variable using `&`.   Alternative, use `*` to add a categorical variable and its interaction with a continuous variable. Variables must be of type CategoricalArray (use `categorical` to convert a variable to a `CategoricalArray`).
* `vcov` : Vcov formula. Default to `simple`. `robust` and `cluster()` are also implemented
* `weights`: Weight variable. Corresponds to analytical weights
* `subset` : Expression of the form State .>= 30

### Returns
* `::Model` : a Model struct


### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StateCategorical] =  categorical(df[:State])
df[:YearCategorical] =  categorical(df[:Year])
reg(df, @model(Sales ~ NDI, weights = Pop))
@model(Sales ~ NDI, fe = StateCategorical, vcov = robust)
@model(Sales ~ NDI, fe = StateCategorical + YearCategorical, weights = Pop, vcov = cluster(StateCategorical)

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


