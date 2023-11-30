# create the matrix X'X
function crossprod(xs...)
    idx = vcat(0, cumsum(size(x, 2) for x in xs))
    XX = zeros(idx[end], idx[end])
    for i in 1:length(xs)
        for j in i:length(xs)
            XX[(idx[i]+1):idx[i+1], (idx[j]+1):idx[j+1]] .= xs[i]' * xs[j]
        end  
    end
    return Symmetric(XX, :U)
end

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

## Returns base of [A B C ...]
## Important: it must be the case that it returns the in order, that is [A B A] returns [true true false] not [false true true]
function basis(xs...; has_intercept = false)
    invXX = invsym!(crossprod(xs...); has_intercept = has_intercept, setzeros = true)
    return diag(invXX) .< 0
end

function getcols(X::AbstractMatrix,  basecolX::AbstractVector)
    sum(basecolX) == size(X, 2) ? X : X[:, basecolX]
end


function ls_solve(X, y::AbstractVector)
    Xy = crossprod(X, y)
    invsym!(Xy, diagonal = 1:size(X, 2))
    return Xy[1:size(X, 2),end]
end

function ls_solve(X, Y::AbstractMatrix)
    XY = crossprod(X, Y)
    invsym!(XY, diagonal = 1:size(X, 2))
    return XY[1:size(X, 2),(end-size(Y, 2)+1):end]
end

##############################################################################
# Auxiliary functions to find columns of exogeneous, endogenous and IV variables
##############################################################################

function find_cols_exo(n_exo)
    2:n_exo+1
end
function find_cols_endo(n_exo, n_endo)
    n_exo+2:n_exo+n_endo+1
end
function find_cols_z(n_exo, n_endo, n_z)
    n_exo+n_endo+2:n_exo+n_endo+n_z+1
end