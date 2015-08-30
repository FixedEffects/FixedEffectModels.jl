
##############################################################################
##
## Fe and FixedEffectSlope
##
##############################################################################
abstract AbstractFixedEffect 

type FixedEffectIntercept{R, W <: AbstractVector{Float64}} <: AbstractFixedEffect
    refs::Vector{R}        # Refs corresponding to the refs field of the original PooledDataVector
    sqrtw::W               # weights
    scale::Vector{Float64} # 1/(sum of scale) within each group
    name::Symbol           # Name of variable in the original dataframe
    id::Symbol
end

type FixedEffectSlope{R, W <: AbstractVector{Float64}} <: AbstractFixedEffect
    refs::Vector{R}              # Refs corresponding to the refs field of the original PooledDataVector
    sqrtw::W                     # weights
    scale::Vector{Float64}       # 1/(sum of weights * x) for each group
    interaction::Vector{Float64} # the continuous interaction 
    name::Symbol                 # Name of factor variable in the original dataframe
    interactionname::Symbol      # Name of continuous variable in the original dataframe
    id::Symbol
end

# Constructors from vectors
function FixedEffectIntercept{R}(refs::Vector{R}, 
                                 l::Int, 
                                 sqrtw::AbstractVector{Float64}, 
                                 name::Symbol, 
                                 id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd  for i in 1:length(refs)
        scale[refs[i]] += abs2(sqrtw[i])
    end
    @inbounds @simd  for i in 1:length(scale)
        scale[i] = scale[i] > 0 ? (1.0 / scale[i]) : zero(Float64)
    end
    FixedEffectIntercept(refs, sqrtw, scale, name, id)
end

function FixedEffectSlope{R}(refs::Vector{R}, 
                             l::Int, 
                             sqrtw::AbstractVector{Float64}, 
                             interaction::Vector{Float64}, 
                             name::Symbol, 
                             interactionname::Symbol, 
                             id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2((interaction[i] * sqrtw[i]))
    end
    @inbounds @simd for i in 1:length(scale)
        scale[i] = scale[i] > 0 ? (1.0 / scale[i]) : zero(Float64)
    end
    FixedEffectSlope(refs, sqrtw, scale, interaction, name, interactionname, id)
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    if a.args[1] == :&
        id = convert(Symbol, "$(a.args[2])x$(a.args[3])")
        if (typeof(df[a.args[2]]) <: PooledDataVector) && !(typeof(df[a.args[3]]) <: PooledDataVector)
            f = df[a.args[2]]
            x = convert(Vector{Float64}, df[a.args[3]])
            return FixedEffectSlope(f.refs, length(f.pool), sqrtw, x, a.args[2], a.args[3], id)
        elseif (typeof(df[a.args[3]]) <: PooledDataVector) && !(typeof(df[a.args[2]]) <: PooledDataVector)
            f = df[a.args[3]]
            x = convert(Vector{Float64}, df[a.args[2]])
            return FixedEffectSlope(f.refs, length(f.pool), sqrtw, x, a.args[3], a.args[2], id)
        else
            error("& is not of the form factor & nonfactor")
        end
    else
        error("Formula should be composed of & and symbols")
    end
end

function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    if typeof(df[a]) <: PooledDataVector
        return FixedEffectIntercept(df[a].refs, length(df[a].pool), sqrtw, a, a)
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

function demean!{R, W <: Ones}(x::AbstractVector{Float64},
                              fe::FixedEffectIntercept{R, W}, 
                              means::Vector{Float64})
    fill!(means, zero(Float64))
    @inbounds @simd for i in 1:length(x)
         means[fe.refs[i]] += x[i] 
    end
    @inbounds @simd for i in 1:length(fe.scale)
         means[i] *= fe.scale[i] 
    end
    @inbounds @simd for i in 1:length(x)
         x[i] -= means[fe.refs[i]] 
    end
end

function demean!{R, W}(x::AbstractVector{Float64},
                       fe::FixedEffectIntercept{R, W}, 
                       means::Vector{Float64})
    fill!(means, zero(Float64))
    @inbounds @simd for i in 1:length(x)
         means[fe.refs[i]] += x[i] * fe.sqrtw[i]
    end
    @inbounds @simd for i in 1:length(fe.scale)
         means[i] *= fe.scale[i] 
    end
    @inbounds @simd for i in 1:length(x)
         x[i] -= means[fe.refs[i]] * fe.sqrtw[i]
    end
end

function demean!{R, W <: Ones}(x::AbstractVector{Float64}, 
                              fe::FixedEffectSlope{R, W}, 
                              means::Vector{Float64})
    fill!(means, zero(Float64))
    @inbounds @simd for i in 1:length(x)
         means[fe.refs[i]] += x[i] * fe.interaction[i] 
    end
    @inbounds @simd for i in 1:length(fe.scale)
         means[i] *= fe.scale[i] 
    end
    @inbounds @simd for i in 1:length(x)
         x[i] -= means[fe.refs[i]] * fe.interaction[i] 
    end
end

function demean!{R, W}(x::AbstractVector{Float64}, 
                       fe::FixedEffectSlope{R, W}, 
                       means::Vector{Float64})
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

function demean!(x::AbstractVector{Float64}, 
                 iterationsv::Vector{Int}, 
                 convergedv::Vector{Bool},
                 fes::Vector{AbstractFixedEffect};
                 maxiter::Int = 1000,
                 tol::Float64 = 1e-8)
    # allocate array of means for each factor
    dict = Dict{AbstractFixedEffect, Vector{Float64}}()
    for fe in fes
        dict[fe] = zeros(Float64, length(fe.scale))
    end
    iterations = maxiter
    converged = false
    if length(fes) == 1 && typeof(fes[1]) <: FixedEffectIntercept
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
    return x
end

function demean!(X::Matrix{Float64}, 
                 iterations::Vector{Int}, 
                 converged::Vector{Bool}, 
                 fes::Vector{AbstractFixedEffect}; 
                 maxiter::Int = 1000, 
                 tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        demean!(slice(X, :, j), iterations, converged, fes, maxiter = maxiter, tol = tol)
    end
end

function demean(x::DataVector{Float64}, 
                fes::Vector{AbstractFixedEffect}; 
                maxiter::Int = 1000, 
                tol::Float64 = 1e-8)
    x = convert(Vector{Float64}, x)
    iterations = Int[]
    converged = Bool[]
    demean!(x, iterations, converged, fes, maxiter = maxiter, tol = tol)
    return x, iterations, converged
end

function demean!(::Array,
                 ::Vector{Int}, 
                 ::Vector{Bool}, 
                 ::Nothing; 
                 maxiter::Int = 1000, 
                 tol::Float64 = 1e-8)
    nothing
end
