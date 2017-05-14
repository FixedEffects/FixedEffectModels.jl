##############################################################################
##
## VcovData stores data you need to compute errors
##
##############################################################################

struct VcovData{T, N} 
    regressors::Matrix{Float64}       # X
    crossmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    df_residual::Int
end

VcovDataVector{T} = VcovData{T, 1} 
VcovDataMatrix{T} = VcovData{T, 2} 




##############################################################################
##
## VcovDataMethod stores data you need to compute errors
##
##############################################################################
abstract type AbstractVcovFormula end
macro vcov()
    :(VcovSimpleFormula())
end
macro vcov(ex)
    if isa(ex, Symbol)
        return Expr(:macrocall, Symbol("@vcov$(ex)"))
    elseif ex.head == :call
        return Expr(:macrocall, Symbol("@vcov$(ex.args[1])"), (esc(ex.args[i]) for i in 2:length(ex.args))...)
    else
        throw("vcov has error")
    end
end

allvars(::AbstractVcovFormula) = Symbol[]

abstract type AbstractVcovMethod end
sandwich(H, S::Matrix{Float64}) = H \ S * inv(H)
df_FStat(::AbstractVcovMethod, x::VcovData, hasintercept::Bool) = x.df_residual - hasintercept
