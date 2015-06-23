module FixedEffectModels
import StatsBase: coef, nobs, coeftable, vcov, residuals, var
import GLM: df_residual
import DataFrames: allvars, Terms
import Distributions: TDist

export group, demean!, demean, reg,
# export type
RegressionResult,
AbstractVce, VceSimple, VceWhite, VceHac, VceCluster,
AbstractVceData, VceData, VceDataHat,
FactorModel

include("utils.jl")
include("vcov.jl")
include("areg.jl")
include("demean.jl")


# A type of fe Model

#immutable FeModel 
#	formula::Formula
#	fe::Union(Vector, nothing)
#end
#FeModel(f::Formula) = FeModel(f, nothing)
#
#
#immutable InteractiveFeModel
#	formula::Formula
#	fe::Union(Vector, nothing)
#	id::Symbol
#	time::Symbol
#	dimension::Int64
#end
#InteractiveFeModel(f::Formula, id::Symbol, time::Symbol, dimension::Int64) = InteractiveFeModel(f, nothing, id, time, dimension)


# A type that stores results of fe model
type RegressionResult <: RegressionModel
	coef::Vector{Float64}
	vcov::Matrix{Float64}

	r2::Float64
	r2_a::Float64
	F::Float64
	nobs::Int64
	df_residual::Int64

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

function StatsBase.residuals(x::RegressionResult, df::AbstractDataFrame)
	mf = ModelFrame(x.terms, df)
	model_response(mf) -  ModelMatrix(mf).m * beta
end

function StatsBase.model_response(x::RegressionResult, df::AbstractDataFrame)
    model_response(ModelFrame(newTerms, df))
end

function Base.show(io::IO, x::RegressionResult) 
	show(io, coeftable(x))
end


function StatsBase.coeftable(x::RegressionResult) 
    title = "Fixed Effect Model"
    top = ["Dependent variable" string(x.yname);
            "Number of obs" sprint(showcompact, x.nobs);
            "Degree of freedom" sprint(showcompact, x.nobs-x.df_residual);
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.r2_a);
            "F Statistics" sprint(showcompact, x.F);]
	cc = coef(x)
	se = stderr(x)
	tt = cc ./ se
	coefnames = x.coefnames
	# put (intercept) last
	if coefnames[1] == symbol("(Intercept)") 
		newindex = vcat(2:length(cc), 1)
		cc = cc[newindex]
		se = se[newindex]
		coefnames = coefnames[newindex]
	end
    scale = quantile(TDist(df_residual(x)), 1 - (1-0.95)/2)
    CoefTable2(hcat(cc, se, tt, ccdf(FDist(1, df_residual(x)), abs2(tt)), cc -  scale * se, cc + scale * se),
              ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
              ["$(coefnames[i])" for i = 1:length(cc)], 4, title, top)
end

## Nms are the coefficient names, corresponding to rows in the table
## Vector and vectornms are things to print before the table
## Title is the title of the table
type CoefTable2
    mat::Matrix
    colnms::Vector
    rownms::Vector
    pvalcol::Integer
    title::String
    top::Matrix{String}
    function CoefTable2(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0, title::String = "", top::Matrix = Any[])
        nr,nc = size(mat)
        0 <= pvalcol <= nc || error("pvalcol = $pvalcol should be in 0,...,$nc]")
        length(colnms) in [0,nc] || error("colnms should have length 0 or $nc")
        length(rownms) in [0,nr] || error("rownms should have length 0 or $nr")
        length(top) == 0 || (size(top, 2) == 2 || error("top should have 2 columns"))
        new(mat,colnms,rownms,pvalcol, title, top)
    end
end

## format numbers in the p-value column
function format_scientific(pv::Number)
    return @sprintf("%.3f", pv)
end

function show(io::IO, ct::CoefTable2)
    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms; pvc = ct.pvalcol; title = ct.title; top = ct.top
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3)::String for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(showcompact,mat[i,j]) for i in 1:nr, j in 1:nc]
    if pvc != 0                         # format the p-values column
        for i in 1:nr
            str[i,pvc] = format_scientific(mat[i,pvc])
        end
    end
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i,j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    totalwidth = sum(widths) + rnwidth
    if length(title) > 0 
        halfwidth = div(totalwidth - length(title), 2) 
        println(io, " " ^ halfwidth * string(title) * " " ^ halfwidth)
    end
    if length(top) > 0
        for i in size(top, 1)
            top[i, 1] = top[i, 1] * ":"
        end
        println(io, "=" ^totalwidth)
        halfwidth = div(totalwidth, 2) - 1 
        interwidth = 2 +  mod(totalwidth, 2)
        for i in 1:(div(size(top, 1) - 1, 2)+1)
            print(io, top[2*i-1, 1])
            print(io, lpad(top[2*i-1, 2], halfwidth - length(top[2*i-1, 1])))
            print(io, " " ^interwidth)
            if size(top, 1) >= 2*i
                print(io, top[2*i, 1])
                print(io, lpad(top[2*i, 2], halfwidth - length(top[2*i, 1])))
            end
            println(io)
        end
    end
    println("=" ^totalwidth)
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    println("-" ^totalwidth)
    for i = 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
    println("=" ^totalwidth)
end


end