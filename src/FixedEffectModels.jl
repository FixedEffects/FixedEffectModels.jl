
module FixedEffectModels

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
	@eval Base.Experimental.@optlevel 1
end

using DataFrames
using FixedEffects
using LinearAlgebra
using Printf
using Reexport
using Statistics
using StatsBase
using StatsFuns
@reexport using StatsModels
using Tables
using Vcov

include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")
include("utils/formula.jl")
include("FixedEffectModel.jl")
include("fit.jl")
include("partial_out.jl")

include("precompile.jl")
_precompile_()

# Export from StatsBase
export coef, coefnames, coeftable, responsename, vcov, stderror, nobs, dof_residual, r2, adjr2, islinear, deviance, rss, mss, confint, predict, residuals

export reg,
partial_out,
fe,
FixedEffectModel,
has_iv,
has_fe,
Vcov

end 