
module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
import Base: size, copyto!, getindex, length, fill!, eltype, length, view, adjoint
import LinearAlgebra: mul!, rmul!, norm, Matrix, Diagonal, cholesky, cholesky!, Symmetric, Hermitian, rank, dot, eigen, axpy!, svd, I, Adjoint, diag, qr
import LinearAlgebra.BLAS: gemm!
import Statistics: mean, quantile
import Printf: @sprintf
import Distributions: ccdf, TDist, FDist, Chisq
import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing, groupby, groupindices
import Combinatorics: combinations
using CategoricalArrays
using FillArrays
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderror, confint, fit, CoefTable, dof_residual, r2, adjr2, deviance, mss, rss, islinear, response, modelmatrix
import StatsModels: @formula,  FormulaTerm, Term, ModelFrame, ModelMatrix, coefnames, columntable, missing_omit, termvars
using Reexport
@reexport using StatsBase
@reexport using FixedEffects

##############################################################################
##
## Exported methods and types
##
##############################################################################

export reg,
partial_out,
allvars,
fes,
WeightFormula,

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
include("utils/weights.jl")
include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")

include("formula/model.jl")
include("formula/formula_iv.jl")
include("formula/formula_fe.jl")


include("RegressionResult.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovrobust.jl")
include("vcov/vcovcluster.jl")
include("vcov/utils.jl")

include("reg.jl")
include("partial_out.jl")



end  # module FixedEffectModels
