
##############################################################################
##
## Type FixedEffectModel
##
##############################################################################

struct FixedEffectModel <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_schema

    nobs::Int64             # Number of observations
    dof_residual::Int64      # degrees of freedoms

    rss::Float64            # Sum of squared residuals
    tss::Float64            # Total sum of squares
    r2::Float64             # R squared
    adjr2::Float64          # R squared adjusted

    F::Float64              # F statistics
    p::Float64              # p value for the F statistics

    # for FE
    feformula::Union{Symbol, Expr, Nothing}     
    iterations::Union{Int, Nothing}         # Number of iterations
    converged::Union{Bool, Nothing}         # Has the demeaning algorithm converged?
    r2_within::Union{Float64, Nothing}      # within r2 (with fixed effect

    # for IV
    F_kp::Union{Float64, Nothing}           # First Stage F statistics KP
    p_kp::Union{Float64, Nothing}           # First Stage p value KP
end

has_iv(x::FixedEffectModel) = x.F_kp !== nothing
has_fe(x::FixedEffectModel) = x.feformula !== nothing

function fit(::Type{FixedEffectModel}, m::Model, df::AbstractDataFrame; kwargs...)
    reg(m, df; kwargs...)
end

# Check API at  https://github.com/JuliaStats/StatsBase.jl/blob/11a44398bdc16a00060bc6c2fb65522e4547f159/src/statmodels.jl
# fields
coef(x::FixedEffectModel) = x.coef
coefnames(x::FixedEffectModel) = x.coefnames
vcov(x::FixedEffectModel) = x.vcov
nobs(x::FixedEffectModel) = x.nobs
dof_residual(x::FixedEffectModel) = x.dof_residual
r2(x::FixedEffectModel) = x.r2
adjr2(x::FixedEffectModel) = x.adjr2
islinear(x::FixedEffectModel) = true
deviance(x::FixedEffectModel) = x.tss
rss(x::FixedEffectModel) = x.rss
mss(x::FixedEffectModel) = deviance(x) - rss(x)

function confint(x::FixedEffectModel)
    scale = quantile(TDist(dof_residual(x)), 1 - (1-0.95)/2)
    se = stderror(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end

# predict, residuals, modelresponse
function predict(x::FixedEffectModel, df::AbstractDataFrame)
    has_fe(x) && error("predict is not defined for fixed effect models. To access the fixed effects, run `reg` with the option save = true, and access fixed effects with `fes()`")
    cols, nonmissings = missing_omit(columntable(df), MatrixTerm(x.formula_schema.rhs))
    new_x = modelmatrix(x.formula_schema, cols)
    out = Vector{Union{Float64, Missing}}(missing, size(df, 1))
    out[nonmissings] = new_x * x.coef
    return out
end

function residuals(x::FixedEffectModel, df::AbstractDataFrame)
    if !has_fe(x)
        cols, nonmissings = missing_omit(columntable(df), x.formula_schema)
        new_x = modelmatrix(x.formula_schema, cols)
        y = response(x.formula_schema, df)
        out = Vector{Union{Float64, Missing}}(missing,  size(df, 1))
        out[nonmissings] = y -  new_x * x.coef
        return out
    else
        size(x.augmentdf, 2) == 0 && error("To access residuals in a fixed effect regression,  run `reg` with the option save = true, and then access residuals with `residuals()`")
       residuals(x)
   end
end

function residuals(x::FixedEffectModel)
    !has_fe(x) && error("To access residuals,  use residuals(x, df::AbstractDataFrame")
    x.augmentdf.residuals
end
   
function fes(x::FixedEffectModel)
   !has_fe(x) && error("fes is not defined for fixed effect models without fixed effects")
   x.augmentdf[!, 2:size(x.augmentdf, 2)]
end

##############################################################################
##
## Display Result
##
##############################################################################

function coeftable(x::FixedEffectModel)
    ctitle = title(x)
    ctop = top(x)
    cc = coef(x)
    se = stderror(x)
    coefnms = coefnames(x)
    conf_int = confint(x)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    CoefTable2(
        hcat(cc, se, tt, ccdf.(Ref(FDist(1, dof_residual(x))), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4, ctitle, ctop)
end

function Base.show(io::IO, x::FixedEffectModel)
    show(io, coeftable(x))
end

function title(x::FixedEffectModel)
    iv = has_iv(x)
    fe = has_fe(x)
    if !iv & !fe
        return "Linear Model"
    elseif iv & !fe
        return "IV Model"
    elseif !iv & fe
        return "Fixed Effect Model"
    elseif iv & fe
        return "IV Fixed Effect Model"
    end
end

function top(x::FixedEffectModel)
    out = [
            "Number of obs" sprint(show, nobs(x), context = :compact => true);
            "Degrees of freedom" sprint(show, nobs(x) - dof_residual(x), context = :compact => true);
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.adjr2);
            "F Statistic" sprint(show, x.F, context = :compact => true);
            "p-value" format_scientific(x.p);
            ]
    if has_iv(x)
        vcat(out, 
            ["First Stage F-stat (KP)" sprint(show, x.F_kp, context = :compact => true);
            "First Stage p-val (KP)" format_scientific(x.p_kp);
            ])
    end
    if has_fe(x)
        vcat(out, 
            ["R2 within" format_scientific(x.r2_within);
           "Iterations" sprint(show, x.iterations, context = :compact => true);
             "Converged" sprint(show, x.converged, context = :compact => true);
             ])
    end
    return out
end


## Coeftalble2 is a modified Coeftable allowing for a top String matrix displayed before the coefficients.
## Pull request: https://github.com/JuliaStats/StatsBase.jl/pull/119

struct CoefTable2
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

function Base.show(io::IO, ct::CoefTable2)
    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms;
    pvc = ct.pvalcol; title = ct.title;   top = ct.top
    if length(rownms) == 0
        rownms = AbstractString[lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    if length(rownms) > 0
        rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
        else
            # if only intercept, rownms is empty collection, so previous would return error
        rnwidth = 4
    end
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(show, mat[i,j]; context=:compact => true) for i in 1:nr, j in 1:nc]
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
    println(io,"=" ^totalwidth)
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    println(io,"-" ^totalwidth)
    for i in 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
    println(io,"=" ^totalwidth)
end


