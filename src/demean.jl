
##############################################################################
##
## FixedEffect
##
##############################################################################

type FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ w * interaction^2) within each group
    interaction::I          # the continuous interaction 
    factorname::Symbol      # Name of factor variable 
    interactionname::Symbol # Name of continuous variable in the original dataframe
    id::Symbol              # Name of new variable if save = true
end

# Constructors the scale vector
function FixedEffect{R <: Integer}(
    refs::Vector{R}, l::Int, sqrtw::AbstractVector{Float64}, 
    interaction::AbstractVector{Float64}, factorname::Symbol, 
    interactionname::Symbol, id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2((interaction[i] * sqrtw[i]))
    end
    @inbounds @simd for i in 1:l
        scale[i] = scale[i] > 0 ? (1.0 / scale[i]) : zero(Float64)
    end
    FixedEffect(refs, sqrtw, scale, interaction, factorname, interactionname, id)
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    if a.args[1] == :&
        id = convert(Symbol, "$(a.args[2])x$(a.args[3])")
        if (typeof(df[a.args[2]]) <: PooledDataVector) && !(typeof(df[a.args[3]]) <: PooledDataVector)
            f = df[a.args[2]]
            x = convert(Vector{Float64}, df[a.args[3]])
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[2], a.args[3], id)
        elseif (typeof(df[a.args[3]]) <: PooledDataVector) && !(typeof(df[a.args[2]]) <: PooledDataVector)
            f = df[a.args[3]]
            x = convert(Vector{Float64}, df[a.args[2]])
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[3], a.args[2], id)
        else
            error("Exp $(a) should be of the form factor&nonfactor")
        end
    else
        error("Exp $(a) should be of the form factor&nonfactor")
    end
end

function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    v = df[a]
    if typeof(v) <: PooledDataVector
        return FixedEffect(v.refs, length(v.pool), sqrtw, Ones(length(v)), a, :none, a)
    else
        error("$(a) is not a pooled data array")
    end
end

##############################################################################
##
## Demean algorithm
## http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf
##
##############################################################################

function demean!{R, W, I}(
    x::AbstractVector{Float64}, fe::FixedEffect{R, W, I}, means::Vector{Float64})
    fill!(means, zero(Float64))
    @inbounds @simd for i in 1:length(x)
         means[fe.refs[i]] += x[i] * fe.interaction[i] * fe.sqrtw[i]
    end
    @inbounds @simd for i in 1:length(fe.scale)
         means[i] *= fe.scale[i] 
    end
    @inbounds @simd for i in 1:length(x)
         x[i] -= means[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end

function demean!(x::AbstractVector{Float64}, iterationsv::Vector{Int}, 
                 convergedv::Vector{Bool}, fes::Vector{FixedEffect};
                 maxiter::Int = 1000, tol::Float64 = 1e-8)
    # allocate array of means for each factor
    dict = Dict{FixedEffect, Vector{Float64}}()
    for fe in fes
        dict[fe] = zeros(Float64, length(fe.scale))
    end
    iterations = maxiter
    converged = false
    # save on one iteration for very common case
    if length(fes) == 1 && typeof(fes[1].interaction) <: Ones
        converged = true
        iterations = 1
        maxiter = 1
    end
    delta = 1.0
    olx = similar(x)
    for iter in 1:maxiter
        @inbounds @simd for i in 1:length(x)
            olx[i] = x[i]
        end
        for fe in fes
            demean!(x, fe, dict[fe])
        end
        if _chebyshev(x, olx, tol)
            converged = true
            iterations = iter
            break
        end
    end
    push!(iterationsv, iterations)
    push!(convergedv, converged)
end

function demean!(X::Matrix{Float64}, iterations::Vector{Int}, converged::Vector{Bool}, 
                 fes::Vector{FixedEffect}; maxiter::Int = 1000, tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        demean!(slice(X, :, j), iterations, converged, fes, maxiter = maxiter, tol = tol)
    end
end

function demean(x::DataVector{Float64},fes::Vector{FixedEffect}; 
                maxiter::Int = 1000, tol::Float64 = 1e-8)
    x = convert(Vector{Float64}, x)
    iterations = Int[]
    converged = Bool[]
    demean!(x, iterations, converged, fes, maxiter = maxiter, tol = tol)
    return x, iterations, converged
end

function demean!(::Array, ::Vector{Int}, ::Vector{Bool}, ::Nothing; 
                 maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end
