
module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Base
using LinearAlgebra
using Statistics
using Printf
using Distributions
using DataFrames
using Combinatorics
using CategoricalArrays
using FillArrays
using Reexport
@reexport using StatsBase
using StatsModels
using FixedEffects

if !isdefined(FixedEffects, :AbstractFixedEffectSolver)
	AbstractFixedEffectSolver{T} = AbstractFixedEffectMatrix{T}
end
##############################################################################
##
## Exported methods and types
##
##############################################################################

export reg,
partial_out,
allvars,
fe,
fes,
WeightFormula,

FixedEffectModel,
has_iv,
has_fe,
AbstractVcov,
VcovSimple,
VcovRobust,
VcovCluster,
Vcov,

AbstractVcovMethod,
VcovMethod,
VcovSimpleMethod,
VcovWhiteMethod,
VcovClusterMethod,

vcov!,
shat!,
VcovData,

ModelTerm,
@model

##############################################################################
##
## Load files
##
##############################################################################
include("utils/weights.jl")
include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")

include("formula/model.jl")
include("formula/formula_iv.jl")
include("formula/formula_fe.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovrobust.jl")
include("vcov/vcovcluster.jl")
include("vcov/utils.jl")

include("FixedEffectModel.jl")
include("fit.jl")
include("partial_out.jl")

# precompile hint
df = DataFrame(y = [1, 1], x =[1, 2], id = categorical([1, 1]))
reg(df, @model(y ~ x + fe(id)))
reg(df, @model(y ~ x, vcov = cluster(id)))

end  # module FixedEffectModels
