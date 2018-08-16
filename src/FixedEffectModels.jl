
module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
import Base: size, copyto!, getindex, length, fill!, eltype, length, view, adjoint
import LinearAlgebra: mul!, rmul!, norm, Matrix, Diagonal, cholesky, cholesky!, Symmetric, Hermitian, rank, dot, eigen, axpy!, svd, I, Adjoint, diag, qr
import LinearAlgebra.BLAS: gemm!
#to suppress
import Statistics: mean, quantile
import Distributed: pmap
import Printf: @sprintf
if Base.USE_GPL_LIBS
    import SparseArrays: SparseMatrixCSC, sparse
end
import Distributions: ccdf, TDist, FDist, Chisq
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!
import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing
import Combinatorics: combinations
using Reexport
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderror, confint, fit, CoefTable, df_residual, dof_residual, r2, adjr2
@reexport using StatsBase
import StatsModels: @formula,  Formula, ModelFrame, ModelMatrix, Terms, coefnames, evalcontrasts, check_non_redundancy!
##############################################################################
##
## Exported methods and types 
##
##############################################################################

export group, 
reg,
partial_out,
residualize!,
getfe!,
decompose_iv!,
allvars,
fes,

WeightFormula,
Ones,
FixedEffect,
FixedEffectProblem,

AbstractRegressionResult,
title,
top,
RegressionResult,
RegressionResultIV,
RegressionResultFE,
RegressionResultFEIV,


AbstractVcovFormula, 
VcovSimpleFormula, 
VcovRobustFormula, 
VcovClusterFormula,
VcovFormula,

AbstractVcovMethod, 
VcovMethod,
VcovSimpleMethod, 
VcovWhiteMethod, 
VcovClusterMethod,

vcov!,
shat!,
VcovData,

Model,
@model

##############################################################################
##
## Load files
##
##############################################################################
include("utils/group.jl")
include("utils/isnested.jl")
include("utils/formula.jl")
include("utils/model.jl")
include("utils/lsmr.jl")
include("utils/basecol.jl")

include("weight/Ones.jl")
include("weight/weight.jl")



include("fixedeffect/FixedEffect.jl")
include("fixedeffect/FixedEffectProblem.jl")
include("fixedeffect/FixedEffectProblem_LSMR.jl")
if Base.USE_GPL_LIBS
    include("fixedeffect/FixedEffectProblem_Factorization.jl")
end

include("RegressionResult.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovrobust.jl")
include("vcov/vcovcluster.jl")
include("vcov/ranktest.jl")

include("reg.jl")
include("partial_out.jl")



end  # module FixedEffectModels
