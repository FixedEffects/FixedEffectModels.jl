module FixedEffectModels
import StatsBase: coef, nobs, coeftable, vcov, residuals, var
import GLM: df_residual
import DataFrames: allvars, Terms
import Distributions: TDist
export group, demean!, demean, reg, regife, RegressionResult, AbstractVCE, VceSimple, VceWhite, VceHac, VceCluster

include("utils.jl")


include("vcov.jl")

include("reg.jl")
include("demean.jl")

include("regife.jl")


# A light type for regression results
type RegressionResult <: RegressionModel
	coef::Vector{Float64}
	vcov::Matrix{Float64}

	r2::Float64
	r2_a::Float64
	F::Float64
	nobs::Int
	df_residual::Int

	coefnames::Vector{Symbol}
	yname::Symbol
	terms::DataFrames.Terms

	esample::Vector{Bool}

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
	if "(Intercept)" in coefnames
		i = findin("(Intercept)", coefnames)
		index = vcat(setdiff(1:length(cc), i), i)
		cc = cc[index]
		se = se[index]
		coefnames = coefnames[index]
	end
   
    scale = quantile(TDist(df_residual(x)), 1 - (1-0.95)/2)
    CoefTable(hcat(cc, se, tt, ccdf(FDist(1, df_residual(x)), abs2(tt)), cc -  scale * se, cc + scale * se),
              ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
              ["$(coefnames[i])" for i = 1:length(cc)], 4)
end




function Base.show(io::IO, x::RegressionResult) 
	print("\n")
	print(      "Dependent variable:        $(x.yname)\n")
	@printf(io, "Number of obs:             %u\n", x.nobs)
	@printf(io, "Degree of freedom:         %u\n", x.nobs-x.df_residual)
	@printf(io, "R2:                        %f\n", x.r2)
	@printf(io, "R2 adjusted:               %f\n", x.r2_a)
	@printf(io, "F Statistics:              %f\n", x.F)
	print("\n")
	show(coeftable(x))
end




end