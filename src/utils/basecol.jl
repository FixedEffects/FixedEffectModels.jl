##############################################################################
##
## Combination behaves like [A B C ...] without forming it
## 
##############################################################################

struct Combination{N, T <: AbstractMatrix}
    A::NTuple{N, T}
    cumlength::Vector{Int}
end

function Combination(A::AbstractMatrix...)
    Combination(A, cumsum([size(x, 2) for x in A]))
end

function Base.size(c::Combination, i)
    if i == 1
        size(c.A[1], 1)
    elseif i == 2
        c.cumlength[end]
    end
end

function Base.view(c::Combination, ::Colon, j)
    index = searchsortedfirst(c.cumlength, j)
    newj = j
    if index > 1
        newj = j - c.cumlength[index-1]
    end
    view(c.A[index], :, newj)
end

##############################################################################
##
## Crossprod computes [A B C ...]' [A B C ...] without forming it
## 
##############################################################################

# Construct [A B C]'[A B C] without generating [A B C]
function crossprod(c::Combination)
    out = Array{Float64}(undef, size(c, 2), size(c, 2))
    cviews = [view(c, :, i) for i in 1:size(c, 2)]
    for j in 1:size(c, 2)
        for i in j:size(c, 2)
            out[i, j] = dot(cviews[j], cviews[i])
        end
    end
    # make symmetric
    for j in 1:size(c, 2), i in 1:(j-1)
        out[i, j] = out[j, i]
    end
    return out
end
crossprod(A::AbstractMatrix) = A'A
crossprod(A::AbstractMatrix...) = crossprod(Combination(A...))

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