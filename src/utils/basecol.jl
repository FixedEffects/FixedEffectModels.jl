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
## Returns base of [A B C ...]
## Important: it must be the case that it returns the in order, that is [A B A] returns [true true false] not [false true true]
## 
##
##############################################################################
# rank(A) == rank(A'A)
function basis(xs::AbstractVector...)
    invXX = invsym!(crossprod(xs...))
    return diag(invXX) .> 0
end

function crossprod(xs::AbstractVector...)
    XX = zeros(length(xs), length(xs))
    for i in 1:length(xs)
        for j in 1:i
            XX[i, j] = xs[i]' * xs[j]
        end  
    end
    for i in 1:length(xs)
        for j in (i+1):length(xs)
            XX[i, j] = XX[j, i]
        end  
    end
    return XX
end


# generalized 2inverse (the one used by Stata)
function invsym!(X::AbstractMatrix)
    # The C value adjusts the check to the relative scale of the variable. The C value is equal to the corrected sum of squares for the variable, unless the corrected sum of squares is 0, in which case C is 1. If you specify the NOINT option but not the ABSORB statement, PROC GLM uses the uncorrected sum of squares instead. The default value of the SINGULAR= option, 107, might be too small, but this value is necessary in order to handle the high-degree polynomials used in the literature to compare regression routin
    tols = max.(diag(X), 1)
    for j in 1:size(X, 1)
        d = X[j,j]
        if abs(d) < tols[j] * sqrt(eps())
            X[j,:] .= 0
            X[:,j] .= 0
        else
            X[j,:] = X[j,:] ./ d
            for i in 1:size(X, 1)
                if i != j
                    X[i,:] .= X[i,:] .- X[i,j] .* X[j,:]
                    X[i,j] = -X[i,j] / d
                end
            end
            X[j,j] = 1 / d
        end
    end
    return X
end

function getcols(X::AbstractMatrix,  basecolX::AbstractVector)
    sum(basecolX) == size(X, 2) ? X : X[:, basecolX]
end
