##############################################################################
##
## Combination behaves like [A B C ...] without forming it
## 
##############################################################################

struct Combination{T} <: AbstractMatrix{T}
    A::Tuple
    cumlength::Vector{Int}
end

function Combination(A::Union{AbstractVector{T}, AbstractMatrix{T}}...) where {T}
    Combination{T}(A, cumsum([size(x, 2) for x in A]))
end

Base.size(c::Combination) = (size(c.A[1], 1), c.cumlength[end])
Base.size(c::Combination, i::Integer) = size(c)[i]

function Base.view(c::Combination, ::Colon, j)
    index = searchsortedfirst(c.cumlength, j)
    newj = index == 1 ? j : j - c.cumlength[index-1]
    view(c.A[index], :, newj)
end

##############################################################################
##
## Crossprod computes [A B C ...]' [A B C ...] without forming it
## 
##############################################################################
crossprod(A::AbstractMatrix) = A'A
function crossprod(A::AbstractMatrix, B::AbstractMatrix)
    u11, u12, u22 = A'A, A'B, B'B
    hvcat(2, u11, u12, 
             u12', u22)
end

function crossprod(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    u11, u12, u13 = A'A, A'B, A'C
    u22, u23 = B'B, B'C
    u33 = C'C
    hvcat(3, u11,  u12,  u13, 
             u12', u22,  u23, 
             u13', u23', u33)
end

##############################################################################
##
## Returns base of [A B C ...]
## Important: it must be the case that it returns the in order, that is [A B A] returns [true true false] not [false true true]
## 
##
##############################################################################
# rank(A) == rank(A'A)
function basecol(X::AbstractMatrix...)
    invXX = invsym!(crossprod(X...))
    return diag(invXX) .> 0
end

# generalized 2inverse (the one used by Stata)
function invsym!(X::AbstractMatrix)
    # SThe C value adjusts the check to the relative scale of the variable. The C value is equal to the corrected sum of squares for the variable, unless the corrected sum of squares is 0, in which case C is 1. If you specify the NOINT option but not the ABSORB statement, PROC GLM uses the uncorrected sum of squares instead.
    tols = max.(diag(X), 1)
    @show tols
    for j in 1:size(X, 1)
        d = X[j,j]
        @show d
        if abs(d) < tols[j] * sqrt(eps())
            X[j,:] .= 0
            X[:,j] .= 0
        else
            X[j,:] = X[j,:] ./ d
            for i in 1:size(X, 1)
                if (i != j)
                    X[i,:] .= X[i,:] .- X[i,j] .* X[j,:]
                    X[i,j] = -X[i,j] / d
                end
            end
            X[j,j] = 1/d
        end
    end
    return X
end



function getcols(X::AbstractMatrix,  basecolX::AbstractVector)
    sum(basecolX) == size(X, 2) ? X : X[:, basecolX]
end
