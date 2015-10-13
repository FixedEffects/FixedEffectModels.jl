# basecol returns a base from a set of matrix
## It avoids to generate the matrix [A B C]
## Simply construct the matrix [A B C ...]'[A B C ....] and compute its rank


# Construct Combination type that behaves as appended [A B C]
type Combination{N}
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

function slice(c::Combination, ::Colon, j)
    index = searchsortedfirst(c.cumlength, j)
    newj = j
    if index > 1
        newj = j - c.cumlength[index-1]
    end
    slice(c.A[index], :, newj)
end

# Construct [A B C]'[A B C] without generating [A B C]
function crossprod{N}(c::Combination{N})
    out = Array(Float64,  size(c, 2), size(c, 2))
    idx = 0
    for j in 1:size(c, 2)
        slicej = slice(c, :, j)
        @inbounds for i in j:size(c, 2)
            idx += 1
            out[i, j] = dot(slicej, slice(c, :, i))
        end
    end
    # make symmetric
    @inbounds for j in 1:size(c, 2), i in 1:(j-1)
        out[i, j] = out[j, i]
    end
    return out
end
crossprod(A::Matrix{Float64}) = A'A
crossprod(A::Matrix{Float64}...) = crossprod(Combination(A...))



# rank(A) == rank(A'A)
function basecol(X::Matrix{Float64}...)
    chol = cholfact!(crossprod(X...), :U, Val{true})
    ipermute!(diag(chol.factors) .> 0, chol.piv)
end

function getcols(X::Matrix{Float64},  basecolX::BitArray{1})
    if sum(basecolX) == size(X, 2)
        return X
    else
        return X[:, basecolX]
    end
end