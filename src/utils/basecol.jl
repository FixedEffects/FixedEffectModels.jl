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

## Returns base of X = [A B C ...]. Takes as input the matrix X'X (actuallyjust its right upper-triangular)
## Important: it must be the case that colinear are first columbs in the bsae in the order of columns
## that is [A B A] returns [true true false] not [false true true]
function basis!(XX::Symmetric; has_intercept = false)
    invXX = invsym!(XX; has_intercept = has_intercept, setzeros = true)
    return diag(invXX) .< 0
end


#solve X \ y. Take as input the matrix [X'X, X'y
#                                        y'X, y'y]
# (but only upper matters)
function ls_solve!(Xy::Symmetric, nx)
    if nx > 0
        invsym!(Xy, diagonal = 1:nx)
        return Xy[1:nx, (nx+1):end]
    else
        return zeros(Float64, 0, size(Xy, 2) - nx)
    end
end
