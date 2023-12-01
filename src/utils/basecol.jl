# generalized 2inverse
#actually return minus the symmetric
function invsym!(X::Symmetric; has_intercept = false, setzeros = false, diagonal = 1:size(X, 2))
    tols = max.(diag(X), 1)
    buffer = zeros(size(X, 1))
    for j in diagonal
        d = X[j,j]
        if setzeros && abs(d) < tols[j] * sqrt(eps())
            X.data[1:j,j] .= 0
            X.data[j,(j+1):end] .= 0
        else
            # used to mimic SAS; now similar to SweepOperators
            copy!(buffer, view(X, :, j))
            Symmetric(BLAS.syrk!('U', 'N', -1/d, buffer, one(eltype(X)), X.data))
            rmul!(buffer, 1 / d)
            @views copy!(X.data[1:j-1,j], buffer[1:j-1])        
            @views copy!(X.data[j, j+1:end], buffer[j+1:end])   
            X[j,j] = - 1 / d
        end
        if setzeros && has_intercept && j == 1
            tols = max.(diag(X), 1)
        end
    end
    return X
end

## Returns base of X = [A B C ...]. Takes as input the matrix X'X
## Important: it must be the case that it returns the in order, that is [A B A] returns [true true false] not [false true true]
function basis(XX; has_intercept = false)
    invXX = invsym!(deepcopy(XX); has_intercept = has_intercept, setzeros = true)
    return diag(invXX) .< 0
end

function getcols(X::AbstractMatrix,  basecolX::AbstractVector)
    sum(basecolX) == size(X, 2) ? X : X[:, basecolX]
end

function getrowscols(XX::AbstractMatrix,  basecolX::AbstractVector)
    sum(basecolX) == size(XX, 2) ? XX : XX[basecolX, basecolX]
end

function ls_solve(Xy, nx)
    if nx > 0
        invsym!(Xy, diagonal = 1:nx)
        return Xy[1:nx, (nx+1):end]
    else
        return zeros(Float64, 0, size(Xy, 2) - nx)
    end
end