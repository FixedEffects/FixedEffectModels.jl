
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
using Combinatorics
using FillArrays
using CategoricalArrays
using DataFrames
using Distributions
using Reexport
@reexport using StatsBase
using StatsModels
using FixedEffects

##############################################################################
##
## Exported methods and types
##
##############################################################################

export reg,
partial_out,
fe,
fes,

FixedEffectModel,
has_iv,
has_fe,

Vcov, # constructor
AbstractVcov,
VcovSimple,
VcovRobust,
VcovCluster,

VcovMethod, # constructor
AbstractVcovMethod,
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
include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")
include("utils/model.jl")
include("utils/formula.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovrobust.jl")
include("vcov/vcovcluster.jl")
include("vcov/utils.jl")

include("FixedEffectModel.jl")
include("fit.jl")
include("partial_out.jl")

# precompile script
df = DataFrame(y = [1, 1], x =[1, 2], id = [1, 1])
reg(df, @model(y ~ x + fe(id)))
reg(df, @model(y ~ x, vcov = cluster(id)))

end  # module FixedEffectModels
