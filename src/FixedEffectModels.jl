
module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using LinearAlgebra
using Statistics
using Printf
using FillArrays
using DataFrames
using Distributions
using Reexport
using Tables
using FixedEffects
using LazyArrays
@reexport using StatsBase
@reexport using StatsModels

##############################################################################
##
## Exported methods and types
##
##############################################################################

export reg,
partial_out,
fe,
FixedEffectModel,
has_iv,
has_fe,
Vcov

##############################################################################
##
## Load files
##
##############################################################################
include("vcov/Vcov.jl")

include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")
include("utils/ranktest.jl")
include("utils/formula.jl")


include("FixedEffectModel.jl")
include("fit.jl")
include("partial_out.jl")


end  # module FixedEffectModels
