
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
    formula_schema::FormulaTerm # Schema for predict
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

has_iv(m::FixedEffectModel) = m.F_kp !== nothing
has_fe(m::FixedEffectModel) = has_fe(m.formula)


# Check API at  https://github.com/JuliaStats/StatsBase.jl/blob/11a44398bdc16a00060bc6c2fb65522e4547f159/src/statmodels.jl
# fields
StatsBase.coef(m::FixedEffectModel) = m.coef
StatsBase.coefnames(m::FixedEffectModel) = m.coefnames
StatsBase.responsename(m::FixedEffectModel) = m.yname
StatsBase.vcov(m::FixedEffectModel) = m.vcov
StatsBase.nobs(m::FixedEffectModel) = m.nobs
StatsBase.dof_residual(m::FixedEffectModel) = m.dof_residual
StatsBase.r2(m::FixedEffectModel) = m.r2
StatsBase.adjr2(m::FixedEffectModel) = m.adjr2
StatsBase.islinear(m::FixedEffectModel) = true
StatsBase.deviance(m::FixedEffectModel) = m.tss
StatsBase.rss(m::FixedEffectModel) = m.rss
StatsBase.mss(m::FixedEffectModel) = deviance(m) - rss(m)


function StatsBase.confint(m::FixedEffectModel; level::Real = 0.95)
    scale = tdistinvcdf(dof_residual(m), 1 - (1 - level) / 2)
    se = stderror(m)
    hcat(m.coef -  scale * se, m.coef + scale * se)
end

# predict, residuals, modelresponse
function StatsBase.predict(m::FixedEffectModel, t)
    # Require DataFrame input as we are using leftjoin and select from DataFrames here
    # Make sure fes are saved
    if has_fe(m) 
        !isempty(m.fe) || throw("No estimates for fixed effects found. Fixed effects need to be estimated using the option save = :fe or :all for prediction to work.")
    end
    ct = StatsModels.columntable(t)
    cols, nonmissings = StatsModels.missing_omit(ct, MatrixTerm(m.formula_schema.rhs))
    Xnew = modelmatrix(m.formula_schema, cols)
    out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(ct)))
    out[nonmissings] = Xnew * m.coef 

    # Join FE estimates onto data and sum row-wise
    if has_fe(m)
        df = DataFrame(t; copycols = false)
        fes = leftjoin(select(df, m.fekeys), unique(m.fe); on = m.fekeys, makeunique = true, matchmissing = :equal)
        fes = combine(fes, AsTable(Not(m.fekeys)) => sum)
        out[nonmissings] .+= fes[nonmissings, 1]
    end

    return out
end

function StatsBase.residuals(m::FixedEffectModel, t)
    if has_fe(m)
         m.residuals !== nothing || throw("To access residuals in a fixed effect regression,  run `reg` with the option save = :residuals, and then access residuals with `residuals()`")
        residuals(m)
    else
        ct = StatsModels.columntable(t)
        cols, nonmissings = StatsModels.missing_omit(ct, MatrixTerm(m.formula_schema.rhs))
        Xnew = modelmatrix(m.formula_schema, cols)
        y = response(m.formula_schema, ct)
        if all(nonmissings)
            out =  y -  Xnew * m.coef
        else
            out = Vector{Union{Float64, Missing}}(missing,  length(Tables.rows(ct)))
            out[nonmissings] = y -  Xnew * m.coef
        end
        return out
    end
end


function StatsBase.residuals(m::FixedEffectModel)
    has_fe(m) || throw("To access residuals,  use residuals(x, t) where t is a Table")
    m.residuals
end

"""
   fe(x::FixedEffectModel; keepkeys = false)

Return a DataFrame with fixed effects estimates.
The output is aligned with the original DataFrame used in `reg`.

### Keyword arguments
* `keepkeys::Bool' : Should the returned DataFrame include the original variables used to defined groups? Default to false
"""

function fe(m::FixedEffectModel; keepkeys = false)
   !has_fe(m) && throw("fe() is not defined for fixed effect models without fixed effects")
   if keepkeys
       m.fe
   else
      m.fe[!, (length(m.fekeys)+1):end]
   end
end


function StatsBase.coeftable(m::FixedEffectModel; level = 0.95)
    cc = coef(m)
    se = stderror(m)
    coefnms = coefnames(m)
    conf_int = confint(m; level = level)
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
        hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(dof_residual(m)), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4)
end


##############################################################################
##
## Display Result
##
##############################################################################

function title(m::FixedEffectModel)
    iv = has_iv(m)
    fe = has_fe(m)
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

function top(m::FixedEffectModel)
    out = [
            "Number of obs" sprint(show, nobs(m), context = :compact => true);
            "Degrees of freedom" sprint(show, nobs(m) - dof_residual(m), context = :compact => true);
            "R2" format_scientific(r2(m));
            "R2 Adjusted" format_scientific(adjr2(m));
            "F-Stat" sprint(show, m.F, context = :compact => true);
            "p-value" format_scientific(m.p);
            ]
    if has_iv(m)
        out = vcat(out, 
            ["F-Stat (First Stage)" sprint(show, m.F_kp, context = :compact => true);
            "p-value (First Stage)" format_scientific(m.p_kp);
            ])
    end
    if has_fe(m)
        out = vcat(out, 
            ["R2 within" format_scientific(m.r2_within);
           "Iterations" sprint(show, m.iterations, context = :compact => true);
             ])
    end
    return out
end


function Base.show(io::IO, m::FixedEffectModel)
    ctitle = title(m)
    ctop = top(m)
    cc = coef(m)
    se = stderror(m)
    yname = responsename(m)
    coefnms = coefnames(m)
    conf_int = confint(m)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    mat = hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(dof_residual(m)), abs2.(tt)), conf_int[:, 1:2])
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

