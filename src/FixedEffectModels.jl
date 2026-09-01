
module FixedEffectModels


using DataFrames
using FixedEffects
using LinearAlgebra
using Printf
using Reexport
using PrecompileTools 
using Statistics
using StatsAPI
using StatsBase
using StatsFuns
@reexport using StatsModels
using Tables
using Vcov

include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")
include("utils/formula.jl")
include("utils/vectorterms.jl")
include("FixedEffectModel.jl")
include("fit.jl")
include("partial_out.jl")

# Export from StatsBase
export coef, coefnames, coeftable, responsename, vcov, stderror, nobs, dof, dof_residual, r2,  r², adjr2, adjr², islinear, deviance, nulldeviance, rss, mss, confint, predict, residuals, fit,
    loglikelihood, nullloglikelihood, dof_fes


export reg,
partial_out,
fe,
FixedEffectModel,
has_iv,
has_fe,
Vcov


@compile_workload begin
    df = DataFrame(x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], x2 = [1.0, 2.0, 4.0, 4.0, 3.0, 5.0], y = [3.0, 4.0, 4.0, 5.0, 1.0, 2.0], id = [1, 1, 2, 2, 3, 3], s = ["a", "a", "b", "b", "a", "b"])
    reg(df, @formula(y ~ x1 + x2))
    reg(df, @formula(y ~ x1 + fe(id)))
    reg(df, @formula(y ~ x1), Vcov.cluster(:id))
    # cover the categorical and interaction term types (the vector-based
    # machinery compiles per term type, not per formula shape)
    reg(df, @formula(y ~ x1 + s + x1&x2 + x1&s))
end




end
