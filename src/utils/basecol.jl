##############################################################################
##
## Combination behaves like [A B C ...] without forming it
## 
##############################################################################

struct Combination{N}
    A::NTuple{N, Matrix{Float64}}
    cumlength::Vector{Int}
end

function Combination(A::Matrix{Float64}...)
    cumlength = cumsum([size(x, 2) for x in A])
    Combination(A, cumlength)
end

function size(c::Combination, i)
    if i == 1
        size(c.A[1], 1)
    elseif i == 2
        c.cumlength[end]
    end
end

function view(c::Combination, ::Colon, j)
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
function crossprod(c::Combination{N}) where {N}
    out = Array{Float64}(undef, size(c, 2), size(c, 2))
    for j in 1:size(c, 2)
        viewj = view(c, :, j)
        for i in j:size(c, 2)
            out[i, j] = dot(viewj, view(c, :, i))
        end
    end
    # make symmetric
    for j in 1:size(c, 2), i in 1:(j-1)
        out[i, j] = out[j, i]
    end
    return out
end
function crossprod(A::Matrix{Float64})
    A'A
end
crossprod(A::Matrix{Float64}...) = crossprod(Combination(A...))
##############################################################################
##
## Returns base of [A B C ...]
## 
## TODO: You could protect against roundoff error by using a controlled sum algorithm (similar to sum_kbn) to compute elements of X'X, then converting to BigFloat before factoring.
##
##
##############################################################################




# rank(A) == rank(A'A)
function basecol(X::Matrix{Float64}...; factorization = :Cholesky)
    cholm = cholesky!(Symmetric(crossprod(X...)), Val(true); tol = -1, check = false)
    r = rank(cholm)
    if size(cholm, 1) > 1
        r = sum(diag(cholm.factors) .> size(X[1],1)^2 * eps(cholm.factors[1]))
    end
    invpermute!(1:size(cholm, 1) .<= r, cholm.piv)
    # used to be 
    # r = rank(cholm) but does not work wiht very high regressors 
    # if does not work, switch to QR RANK
end

function getcols(X::Matrix{Float64},  basecolX::BitArray{1})
    if sum(basecolX) == size(X, 2)
        return X
    else
        return X[:, basecolX]
    end
end