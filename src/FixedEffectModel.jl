
##############################################################################
##
## Type FixedEffectModel
##
##############################################################################

struct FixedEffectModel <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix
    vcov_type::CovarianceEstimator
    nclusters::Union{NamedTuple, Nothing}

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals::Union{AbstractVector, Nothing}
    fe::DataFrame
    fekeys::Vector{Symbol}


    coefnames::Vector       # Name of coefficients
    yname::Union{String, Symbol} # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_predict::FormulaTerm
    contrasts::Dict

    nobs::Int64             # Number of observations
    dof_residual::Int64      # nobs - degrees of freedoms

    rss::Float64            # Sum of squared residuals
    tss::Float64            # Total sum of squares
    r2::Float64             # R squared
    adjr2::Float64          # R squared adjusted

    F::Float64              # F statistics
    p::Float64              # p value for the F statistics

    # for FE
    iterations::Union{Int, Nothing}         # Number of iterations
    converged::Union{Bool, Nothing}         # Has the demeaning algorithm converged?
    r2_within::Union{Float64, Nothing}      # within r2 (with fixed effect

    # for IV
    F_kp::Union{Float64, Nothing}           # First Stage F statistics KP
    p_kp::Union{Float64, Nothing}           # First Stage p value KP
end

has_iv(x::FixedEffectModel) = x.F_kp !== nothing
has_fe(x::FixedEffectModel) = has_fe(x.formula)


# Check API at  https://github.com/JuliaStats/StatsBase.jl/blob/11a44398bdc16a00060bc6c2fb65522e4547f159/src/statmodels.jl
# fields
StatsBase.coef(x::FixedEffectModel) = x.coef
StatsBase.coefnames(x::FixedEffectModel) = x.coefnames
StatsBase.responsename(x::FixedEffectModel) = x.yname
StatsBase.vcov(x::FixedEffectModel) = x.vcov
StatsBase.nobs(x::FixedEffectModel) = x.nobs
StatsBase.dof_residual(x::FixedEffectModel) = x.dof_residual
StatsBase.r2(x::FixedEffectModel) = x.r2
StatsBase.adjr2(x::FixedEffectModel) = x.adjr2
StatsBase.islinear(x::FixedEffectModel) = true
StatsBase.deviance(x::FixedEffectModel) = x.tss
StatsBase.rss(x::FixedEffectModel) = x.rss
StatsBase.mss(x::FixedEffectModel) = deviance(x) - rss(x)


function StatsBase.confint(x::FixedEffectModel; level::Real = 0.95)
    scale = tdistinvcdf(dof_residual(x), 1 - (1 - level) / 2)
    se = stderror(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end

# predict, residuals, modelresponse
function StatsBase.predict(x::FixedEffectModel, df)
    fes = if has_fe(x)
            sum(Matrix(leftjoin(df, unique(x.fe), on = x.fekeys, makeunique = true)[!, end-length(x.fekeys)+1:end]), dims = 2)
        else zeros(nrow(df))
    end
    df = StatsModels.columntable(df)
    formula_schema = apply_schema(x.formula_predict, schema(x.formula_predict, df, x.contrasts), StatisticalModel)
    cols, nonmissings = StatsModels.missing_omit(df, MatrixTerm(formula_schema.rhs))
    new_x = modelmatrix(formula_schema, cols)
    out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(df)))
    out[nonmissings] = new_x * x.coef .+ fes[nonmissings]
    return out
end

function StatsBase.residuals(x::FixedEffectModel, df)
    df = StatsModels.columntable(df)
    if !has_fe(x)
        formula_schema = apply_schema(x.formula_predict, schema(x.formula_predict, df, x.contrasts), StatisticalModel)
        cols, nonmissings = StatsModels.missing_omit(df, formula_schema)
        new_x = modelmatrix(formula_schema, cols)
        y = response(formula_schema, df)
        if all(nonmissings)
            out =  y -  new_x * x.coef
        else
            out = Vector{Union{Float64, Missing}}(missing,  length(Tables.rows(df)))
            out[nonmissings] = y -  new_x * x.coef
        end
        return out
    else
        typeof(x.residuals) == Nothing && throw("To access residuals in a fixed effect regression,  run `reg` with the option save = :residuals, and then access residuals with `residuals()`")
       residuals(x)
   end
end


function StatsBase.residuals(x::FixedEffectModel)
    !has_fe(x) && throw("To access residuals,  use residuals(x, df::AbstractDataFrame")
    x.residuals
end


"""
   fe(x::FixedEffectModel; keepkeys = false)

Return a DataFrame with fixed effects estimates.
The output is aligned with the original DataFrame used in `reg`.

### Keyword arguments
* `keepkeys::Bool' : Should the returned DataFrame include the original variables used to defined groups? Default to false
"""

function fe(x::FixedEffectModel; keepkeys = false)
   !has_fe(x) && throw("fe() is not defined for fixed effect models without fixed effects")
   if keepkeys
       x.fe
   else
      x.fe[!, (length(x.fekeys)+1):end]
   end
end


function StatsBase.coeftable(x::FixedEffectModel; level = 0.95)
    cc = coef(x)
    se = stderror(x)
    coefnms = coefnames(x)
    conf_int = confint(x; level = level)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    CoefTable(
        hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(dof_residual(x)), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4)
end


##############################################################################
##
## Display Result
##
##############################################################################

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

format_scientific(x) = @sprintf("%.3f", x)

function top(x::FixedEffectModel)
    out = [
            "Number of obs" sprint(show, nobs(x), context = :compact => true);
            "Degrees of freedom" sprint(show, nobs(x) - dof_residual(x), context = :compact => true);
            "R2" format_scientific(x.r2);
            "R2 Adjusted" format_scientific(x.adjr2);
            "F-Stat" sprint(show, x.F, context = :compact => true);
            "p-value" format_scientific(x.p);
            ]
    if has_iv(x)
        out = vcat(out,
            ["F-Stat (First Stage)" sprint(show, x.F_kp, context = :compact => true);
            "p-value (First Stage)" format_scientific(x.p_kp);
            ])
    end
    if has_fe(x)
        out = vcat(out,
            ["R2 within" format_scientific(x.r2_within);
           "Iterations" sprint(show, x.iterations, context = :compact => true);
             ])
    end
    return out
end


function Base.show(io::IO, x::FixedEffectModel)
    ctitle = title(x)
    ctop = top(x)
    cc = coef(x)
    se = stderror(x)
    yname = responsename(x)
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
    mat = hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(dof_residual(x)), abs2.(tt)), conf_int[:, 1:2])
    nr, nc = size(mat)
    colnms = ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%"]
    rownms = ["$(coefnms[i])" for i = 1:length(cc)]
    pvc = 4


    # print
    if length(rownms) == 0
        rownms = AbstractString[lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    if length(rownms) > 0
        rnwidth = max(4, maximum(length(nm) for nm in rownms) + 2, length(yname) + 2)
        else
            # if only intercept, rownms is empty collection, so previous would return error
        rnwidth = 4
    end
    rownms = [rpad(nm,rnwidth-1) * "|" for nm in rownms]
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
    if length(ctitle) > 0
        halfwidth = div(totalwidth - length(ctitle), 2)
        println(io, " " ^ halfwidth * string(ctitle) * " " ^ halfwidth)
    end
    if length(ctop) > 0
        for i in 1:size(ctop, 1)
            ctop[i, 1] = ctop[i, 1] * ":"
        end
        println(io, "=" ^totalwidth)
        halfwidth = div(totalwidth, 2) - 1
        interwidth = 2 +  mod(totalwidth, 2)
        for i in 1:(div(size(ctop, 1) - 1, 2)+1)
            print(io, ctop[2*i-1, 1])
            print(io, lpad(ctop[2*i-1, 2], halfwidth - length(ctop[2*i-1, 1])))
            print(io, " " ^interwidth)
            if size(ctop, 1) >= 2*i
                print(io, ctop[2*i, 1])
                print(io, lpad(ctop[2*i, 2], halfwidth - length(ctop[2*i, 1])))
            end
            println(io)
        end
    end
    println(io,"=" ^totalwidth)
    println(io, rpad(string(yname), rnwidth-1) * "|" *
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


##############################################################################
##
## Schema
##
##############################################################################
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema, Mod::Type{FixedEffectModel}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
                StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
end
