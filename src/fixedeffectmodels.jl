module FixedEffectModels

export group, demean!, demean, areg, regife, RegressionResult

include("algo.jl")
include("vcov.jl")

include("areg.jl")
include("demean.jl")
include("regife.jl")

include("group.jl")

# A light type for regression result
type RegressionResult <: RegressionModel
	coef
	vcov
	
	coefnames
	yname

	nobs
	df_residual

	esample

	terms
end

StatsBase.coef(x::RegressionResult) = x.coef
StatsBase.vcov(x::RegressionResult) = x.vcov
StatsBase.nobs(x::RegressionResult) = x.nobs
df_residual(x::RegressionResult) = x.df_residual


function StatsBase.predict(x::RegressionResult, df::AbstractDataFrame)
    # copy terms, removing outcome if present
    newTerms = remove_response(x.terms)
    # create new model frame/matrix
    newX = ModelMatrix(ModelFrame(newTerms, df)).m
    newX * x.beta
end


function StatsBase.coeftable(x::RegressionResult) 
    cc = coef(x)
    se = stderr(x)
    tt = cc ./ se
    coefnames = x.coefnames
    CoefTable(hcat(cc,se,tt,ccdf(FDist(1, df_residual(x)), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              ["$(coefnames[i])" for i = 1:length(cc)], 4)
end


function Base.show(io::IO, x::RegressionResult) 
	print("\n")
	print(      "Dependent variable:        $(x.yname)\n")
	@printf(io, "Number of obs:             %u\n", x.nobs)
	@printf(io, "Degree of freedom:         %u\n", x.nobs-x.def_residual)
	print("\n")
	show(coeftable(x))
end




end