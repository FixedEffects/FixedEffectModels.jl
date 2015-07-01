# An abstract type that stores light regression results 

abstract AbstractRegressionResult <: RegressionModel


# Type for non IV models
type RegressionResult <: AbstractRegressionResult
    coef::Vector{Float64}
    vcov::Matrix{Float64}

    esample::BitVector

    coefnames::Vector
    yname::Symbol
    formula::Formula

    nobs::Int64
    df_residual::Int64

    r2::Float64
    r2_a::Float64
    F::Float64    
    p::Float64

    function RegressionResult(coef::Vector{Float64}, vcov::Matrix{Float64}, esample::BitVector, coefnames::Vector, yname::Symbol, formula::Formula, nobs::Int64, df_residual::Int64, r2::Float64, r2_a::Float64, F::Float64, p::Float64)
        length(coef) == length(coefnames) || error("Coef and coefnames should have the same size")
        length(coef) == size(vcov, 1) || error("Coef and vcov should have the same dimension")
        length(coef) == size(vcov, 2) || error("Coef and vcov should have the same dimension")
        new(coef, vcov, esample, coefnames, yname, formula, nobs, df_residual, r2, r2_a, F, p)
    end
end

# Type for IV models 
type RegressionResultIV <: AbstractRegressionResult
    coef::Vector{Float64}
    vcov::Matrix{Float64}

    esample::BitVector

    coefnames::Vector
    yname::Symbol
    formula::Formula

    nobs::Int64
    df_residual::Int64

    r2::Float64
    r2_a::Float64
    F::Float64    
    p::Float64    
    F_kp::Float64
    p_kp::Float64

    function RegressionResultIV(coef::Vector{Float64}, vcov::Matrix{Float64}, esample::BitVector, coefnames::Vector, yname::Symbol, formula::Formula, nobs::Int64, df_residual::Int64, r2::Float64, r2_a::Float64, F::Float64, p::Float64, F_kp::Float64, p_kp::Float64)
        length(coef) == length(coefnames) || error("Coef and coefnames should have the same size")
        length(coef) == size(vcov, 1) || error("Coef and vcov should have the same dimension")
        length(coef) == size(vcov, 2) || error("Coef and vcov should have the same dimension")
        new(coef, vcov, esample, coefnames, yname, formula, nobs, df_residual, r2, r2_a, F, p, F_kp, p_kp)
    end
end




coef(x::AbstractRegressionResult) = x.coef
coefnames(x::AbstractRegressionResult) = x.coefnames
vcov(x::AbstractRegressionResult) = x.vcov
nobs(x::AbstractRegressionResult) = x.nobs
function confint(x::AbstractRegressionResult) 
    scale = quantile(TDist(df_residual(x)), 1 - (1-0.95)/2)
    se = stderr(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end
df_residual(x::AbstractRegressionResult) = x.df_residual

function predict(x::AbstractRegressionResult, df::AbstractDataFrame)
    rf = x.formula
    (rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
    has_absorb && error("predict is not defined for fixed effect models (yet)")

    (rf, has_instrument, instrument_formula, endo_formula) = decompose_iv!(rf)
    if has_instrument
        if typeof(rf.rhs) == Symbol
            rf.rhs = endo_formula.rhs
        else        
            push!(rf.rhs.args, endo_formula.rhs)
        end
    end

    newTerms = remove_response(Terms(rf))
    mf = ModelFrame(newTerms, df)
    newX = ModelMatrix(mf).m

    out = DataArray(Float64, size(df, 1))
    out[mf.msng] = newX * x.coef
end

function residuals(x::AbstractRegressionResult, df::AbstractDataFrame)
    rf = x.formula
    (rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
    has_absorb && error("predict is not defined for fixed effect models (yet)")

    (rf, has_instrument, instrument_formula, endo_formula) = decompose_iv!(rf)
    if has_instrument
        if typeof(rf.rhs) == Symbol
            rf.rhs = endo_formula.rhs
        else        
            push!(rf.rhs.args, endo_formula.rhs)
        end
    end

    mf = ModelFrame(Terms(rf), df)
    newX = ModelMatrix(mf).m 
    out = DataArray(Float64, size(df, 1))
    out[mf.msng] = model_response(mf) -  newX * x.coef
end


function model_response(x::AbstractRegressionResult, df::AbstractDataFrame)
    f = x.formula
    mf = ModelFrame(Terms(f), df)
    model_response(mf)
end


function Base.show(io::IO, x::AbstractRegressionResult) 
    show(io, coeftable(x))
end


## Coeftalble2 is a modified Coeftable allowing for a top String matrix displayed before the coefficients. 
## Pull request: https://github.com/JuliaStats/StatsBase.jl/pull/119


function coeftable(x::RegressionResult) 
    title = "Fixed Effect Model"
    top = [
            "Number of obs" sprint(showcompact, x.nobs);
            "Degree of freedom" sprint(showcompact, x.nobs-x.df_residual);
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.r2_a);
            "F Statistic" sprint(showcompact, x.F);
            "Prob > F" format_scientific(x.p);
            ]
    cc = coef(x)
    se = stderr(x)
    coefnames = x.coefnames
    conf_int = confint(x)
    # put (intercept) last
    if (coefnames[1] == symbol("(Intercept)")) || (coefnames[1] == "(Intercept)")
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnames = coefnames[newindex]
    end
    tt = cc ./ se
    CoefTable2(hcat(cc, se, tt, ccdf(FDist(1, df_residual(x)), abs2(tt)), conf_int[:, 1], conf_int[:, 2]),
              ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
              ["$(coefnames[i])" for i = 1:length(cc)], 4, title, top)
end


function coeftable(x::RegressionResultIV) 
    title = "Fixed Effect Model"
    top = [
            "Number of obs" sprint(showcompact, x.nobs);
            "Degree of freedom" sprint(showcompact, x.nobs-x.df_residual);
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.r2_a);
            "F Statistic" sprint(showcompact, x.F);
            "Prob > F" format_scientific(x.p);
            "First Stage F-stat (KP)" sprint(showcompact, x.F_kp);
            "First State p-val (KP)" format_scientific(x.p_kp);
            ]
    cc = coef(x)
    se = stderr(x)
    coefnames = x.coefnames
    conf_int = confint(x)
    # put (intercept) last
    if (coefnames[1] == symbol("(Intercept)")) || (coefnames[1] == "(Intercept)")
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnames = coefnames[newindex]
    end
    tt = cc ./ se
    CoefTable2(hcat(cc, se, tt, ccdf(FDist(1, df_residual(x)), abs2(tt)), conf_int[:, 1], conf_int[:, 2]),
              ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
              ["$(coefnames[i])" for i = 1:length(cc)], 4, title, top)
end




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
