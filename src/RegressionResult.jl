##############################################################################
##
## The lightest type that can (i) print table (ii) predict etc
##
##############################################################################

abstract AbstractRegressionResult <: RegressionModel


# fields
coef(x::AbstractRegressionResult) = x.coef
coefnames(x::AbstractRegressionResult) = x.coefnames
vcov(x::AbstractRegressionResult) = x.vcov
nobs(x::AbstractRegressionResult) = x.nobs
df_residual(x::AbstractRegressionResult) = x.df_residual
function confint(x::AbstractRegressionResult) 
    scale = quantile(TDist(x.df_residual), 1 - (1-0.95)/2)
    se = stderr(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end


# predict, residuals, modelresponse
function predict(x::AbstractRegressionResult, df::AbstractDataFrame)
    rf = deepcopy(x.formula)
    secondstage!(rf)

    newTerms = remove_response(Terms(rf))
    mf = ModelFrame(newTerms, df)
    newX = ModelMatrix(mf).m

    out = DataArray(Float64, size(df, 1))
    out[mf.msng] = newX * x.coef
end

function residuals(x::AbstractRegressionResult, df::AbstractDataFrame)
    rf = deepcopy(x.formula)
    secondstage!(rf)

    mf = ModelFrame(Terms(rf), df)
    newX = ModelMatrix(mf).m 
    out = DataArray(Float64, size(df, 1))
    out[mf.msng] = model_response(mf) -  newX * x.coef
end

function model_response(x::AbstractRegressionResult, df::AbstractDataFrame)
    rf = deepcopy(x.formula)
    secondstage!(rf)
    mf = ModelFrame(Terms(rf), df)
    model_response(mf)
end



# Display Results
function title(::AbstractRegressionResult) 
    error("function title has no general method for AbstractRegressionResult")
end

function top(::AbstractRegressionResult)
    error("function top has no general method for AbstractRegressionResult")
end

function coeftable(x::AbstractRegressionResult)
    ctitle = title(x)
    ctop = top(x)
    cc = coef(x)
    se = stderr(x)
    coefnms = coefnames(x)
    conf_int = confint(x)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    CoefTable2(
        hcat(cc, se, tt, ccdf(FDist(1, df_residual(x)), abs2(tt)), conf_int[:, 1], conf_int[:, 2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4, ctitle, ctop)
end

function Base.show(io::IO, x::AbstractRegressionResult) 
    show(io, coeftable(x))
end


## Coeftalble2 is a modified Coeftable allowing for a top String matrix displayed before the coefficients. 
## Pull request: https://github.com/JuliaStats/StatsBase.jl/pull/119

type CoefTable2
    mat::Matrix
    colnms::Vector
    rownms::Vector
    pvalcol::Integer
    title::AbstractString
    top::Matrix{AbstractString}
    function CoefTable2(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0, 
                        title::AbstractString = "", top::Matrix = Any[])
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
    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms; 
    pvc = ct.pvalcol; title = ct.title;   top = ct.top
    if length(rownms) == 0
        rownms = AbstractString[lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(showcompact,mat[i,j]) for i in 1:nr, j in 1:nc]
    if pvc != 0                         # format the p-values column
        for i in 1:nr
            str[i, pvc] = format_scientific(mat[i, pvc])
        end
    end
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i, j])
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
        for i in 1:size(top, 1)
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
    for i in 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
    println("=" ^totalwidth)
end


##############################################################################
##
## Subtypes of Regression Result
##
##############################################################################

type RegressionResult <: AbstractRegressionResult
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::Formula        # Original formula 

    nobs::Int64             # Number of observations
    df_residual::Int64      # degree of freedoms

    r2::Float64             # R squared
    r2_a::Float64           # R squared adjusted
    F::Float64              # F statistics
    p::Float64              # p value for the F statistics
end
title(::RegressionResult) =  "Linear Model"
top(x::RegressionResult) = [
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.r2_a);
            "F Statistic" sprint(showcompact, x.F);
            "p-value" format_scientific(x.p);
            ]


type RegressionResultIV <: AbstractRegressionResult
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::Formula        # Original formula 

    nobs::Int64             # Number of observations
    df_residual::Int64      # degree of freedoms

    r2::Float64             # R squared
    r2_a::Float64           # R squared adjusted
    F::Float64              # F statistics
    p::Float64              # p value for the F statistics

    F_kp::Float64           # First Stage F statistics KP 
    p_kp::Float64           # First Stage p value KP
end

title(::RegressionResultIV) = "IV Model"
top(x::RegressionResultIV) = [
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.r2_a);
            "F-Statistic" sprint(showcompact, x.F);
            "p-value" format_scientific(x.p);
            "First Stage F-stat (KP)" sprint(showcompact, x.F_kp);
            "First State p-val (KP)" format_scientific(x.p_kp);
            ]


type RegressionResultFE <: AbstractRegressionResult
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::Formula        # Original formula 

    nobs::Int64             # Number of observations
    df_residual::Int64      # degree of freedoms

    r2::Float64             # R squared
    r2_a::Float64           # R squared adjusted
    r2_within::Float64
    F::Float64              # F statistics
    p::Float64              # p value for the F statistics

    iterations::Int         # Number of iterations        
    converged::Bool         # Has the demeaning algorithm converged?
end
function predict(::RegressionResultFE, ::AbstractDataFrame)
    error("predict is not defined for fixed effect models.  Run reg with the the option savefe = true")
end
function residuals(::RegressionResultFE, ::AbstractDataFrame)
    error("residuals is not defined for fixed effect models. Use the function partial_out")
end
title(::RegressionResultFE) = "Fixed Effect Model"
top(x::RegressionResultFE) = [ 
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2" format_scientific(x.r2);
            "R2 within" format_scientific(x.r2_within);
            "F-Statistic" sprint(showcompact, x.F);
            "p-value" format_scientific(x.p);
            "Iterations" sprint(showcompact, x.iterations);
            "Converged" sprint(showcompact, x.converged)
            ]

type RegressionResultFEIV <: AbstractRegressionResult
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::Formula        # Original formula 

    nobs::Int64             # Number of observations
    df_residual::Int64      # degree of freedoms

    r2::Float64             # R squared
    r2_a::Float64           # R squared adjusted
    r2_within::Float64
    F::Float64              # F statistics
    p::Float64              # p value for the F statistics
    
    F_kp::Float64           # First Stage F statistics KP 
    p_kp::Float64           # First Stage p value KP

    iterations::Int         # Number of iterations        
    converged::Bool         # Has the demeaning algorithm converged?
end
function predict(::RegressionResultFEIV, ::AbstractDataFrame)
    error("predict is not defined for fixed effect models. Run reg with the the option savefe = true")
end
function residuals(::RegressionResultFEIV, ::AbstractDataFrame)
    error("residuals is not defined for fixed effect models. Use the function partial_out")
end
title(::RegressionResultFEIV) = "Fixed effect IV Model"
top(x::RegressionResultFEIV) = [
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2" format_scientific(x.r2);
            "R2 within" format_scientific(x.r2_within);
            "F Statistic" sprint(showcompact, x.F);
            "p-value" format_scientific(x.p);
            "First Stage F-stat (KP)" sprint(showcompact, x.F_kp);
            "First State p-val (KP)" format_scientific(x.p_kp);
            "Iterations" sprint(showcompact, x.iterations);
            "Converged" sprint(showcompact, x.converged)
            ]






