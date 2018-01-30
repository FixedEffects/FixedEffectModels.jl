



using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])



##############################################################################
##
## Alternative Syntax @reg df y ~ x2 fe(ok) weight(ok) etc
##
##############################################################################

macro subset(ex)
    return ex
end

macro maxiter(ex)
    return ex
end
macro tol(ex)
    return ex
end
macro df_add(ex)
    return ex
end
macro save(ex)
    return ex
end
macro method(ex)
    return ex
end

function make_macro(x)
    x.head == :call || throw("Argument $(x) is not a function call")
    Expr(:kw, x.args[1], Expr(:macrocall, Symbol("@$(x.args[1])"), (esc(x.args[i]) for i in 2:length(x.args))...))
end
macro reg(kw...)
    Expr(:call, :reg, esc(kw[1]), :(@formula $(esc(kw[2]))), (make_macro(kw[i]) for i in 3:length(kw))...)
end

