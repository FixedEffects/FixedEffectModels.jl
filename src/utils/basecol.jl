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
## 
## TODO: You could protect against roundoff error by using a controlled sum algorithm (similar to sum_kbn) to compute elements of X'X, then converting to BigFloat before factoring.
##
##
##############################################################################

# rank(A) == rank(A'A)
function basecol(X::AbstractMatrix...; factorization = :Cholesky)
    cholm = cholesky!(Symmetric(crossprod(X...)), Val(true); tol = -1, check = false)
    r = 0
    if size(cholm, 1) > 0
        r = sum(diag(cholm.factors) .> size(X[1],1)^2 * eps())
        # used to be r = rank(cholm) but does not work wiht very high regressors at the same time as intercept
    end
    invpermute!(1:size(cholm, 1) .<= r, cholm.piv)
end

function getcols(X::AbstractMatrix,  basecolX::AbstractVector)
    sum(basecolX) == size(X, 2) ? X : X[:, basecolX]
end