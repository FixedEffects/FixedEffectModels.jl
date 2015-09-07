
##############################################################################
##
## FixedEffect
##
##############################################################################

type FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ sqrt(w) * interaction) within each group
    value::Vector{Float64}  # fixed effect values
    interaction::I          # the continuous interaction 
    factorname::Symbol      # Name of factor variable 
    interactionname::Symbol # Name of continuous variable in the original dataframe
    id::Symbol              # Name of new variable if save = true
end

# Constructor
function FixedEffect{R <: Integer}(
    refs::Vector{R}, l::Int, sqrtw::AbstractVector{Float64}, 
    interaction::AbstractVector{Float64}, factorname::Symbol, 
    interactionname::Symbol, id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2((interaction[i] * sqrtw[i]))
    end
    @inbounds @simd for i in 1:l
        scale[i] = scale[i] != 0 ? (1.0 / sqrt(scale[i])) : zero(Float64)
    end
    FixedEffect(refs, sqrtw, scale, similar(scale), interaction, factorname, interactionname, id)
end

##############################################################################
##
## Build from DataFrame
##
##############################################################################

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    if a.args[1] == :&
        id = convert(Symbol, "$(a.args[2])x$(a.args[3])")
        if isa(df[a.args[2]], PooledDataVector) && !isa(df[a.args[3]], PooledDataVector)
            f = df[a.args[2]]
            x = convert(Vector{Float64}, df[a.args[3]])
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[2], a.args[3], id)
        elseif isa(df[a.args[3]], PooledDataVector) && !isa(df[a.args[2]], PooledDataVector)
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
## Denote M model matrix of fixed effects
## Conjugate gradient will solve (A'A)X = A'y
## where A = M * diag(1/a_1, 1/a_2... , 1/a_n) 
## We need to define what it means to multiply by A and At 
##
##############################################################################

type FixedEffectModelMatrix <: AbstractMatrix{Float64}
    _::Vector{FixedEffect}
end

function size(mfe::FixedEffectModelMatrix, i::Integer) 
    fes = mfe._
    if i == 1
        return length(fes[1].refs)
    elseif i == 2
        out = zero(Int)
        for fe in fes
            out += length(fe.scale)
        end
        return out
    end
end

function copy!(mfe::FixedEffectModelMatrix, x::AbstractVector{Float64})
   idx = 0
   fes = mfe._
   for fe in fes
       @inbounds @simd for i in 1:length(fe.scale)
           idx += 1
           fe.value[i] = x[idx] * fe.scale[i]
       end
   end
end

function copy!(x::AbstractVector{Float64}, mfe::FixedEffectModelMatrix)
    idx = 0
    fes = mfe._
    for fe in fes
        @inbounds @simd for i in 1:length(fe.scale)
            idx += 1
            x[idx] = fe.value[i] * fe.scale[i]
        end
    end
    return x
end

# Define x -> A * x
function A_mul_B_helper!{R, W, I}(y::AbstractVector{Float64}, fe::FixedEffect{R, W, I})
    @inbounds @simd for i in 1:length(y)
        y[i] += fe.value[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function A_mul_B!(y::AbstractVector{Float64}, mfe::FixedEffectModelMatrix, 
                  x::AbstractVector{Float64})
    copy!(mfe, x)
    fill!(y, zero(Float64))
    fes = mfe._
    for fe in fes
        A_mul_B_helper!(y, fe)
    end
    return y
end

# Define x -> A' * x
function Ac_mul_B_helper!{R, W, I}(fe::FixedEffect{R, W, I}, y::AbstractVector{Float64})
    fill!(fe.value, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        fe.value[fe.refs[i]] += y[i] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function Ac_mul_B!(x::AbstractVector{Float64}, mfe::FixedEffectModelMatrix, 
                   y::AbstractVector{Float64})
    fes = mfe._
    for fe in fes
        Ac_mul_B_helper!(fe, y)
    end
    copy!(x, mfe)
    return x
end


##############################################################################
##
## FixedEffectProblem stores some arrays to solve (A'A)X = A'y multiple times
##
##############################################################################

type FixedEffectProblem <: AbstractMatrix{Float64}
    m::FixedEffectModelMatrix
    q::Vector{Float64}
    s::Vector{Float64}
    p::Vector{Float64}
end

function FixedEffectProblem(m::FixedEffectModelMatrix)
    q = Array(Float64, size(m, 1))
    s = Array(Float64, size(m, 2))
    p = similar(s)
    FixedEffectProblem(m, q, s, p)
end

function FixedEffectProblem(fes::Vector{FixedEffect})
    FixedEffectProblem(FixedEffectModelMatrix(fes))
end

function cgls!(x::Union(AbstractVector{Float64}, Nothing), r::AbstractVector{Float64}, 
               pfe::FixedEffectProblem; tol::Real=1e-10, maxiter::Integer=1000)
    cgls!(x, r, pfe.m, pfe.s, pfe.p, pfe.q; tol = tol, maxiter = maxiter)
end

