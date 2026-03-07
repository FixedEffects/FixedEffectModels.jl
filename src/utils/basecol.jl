"""
    invsym!(X::Symmetric; has_intercept=false, setzeros=false, diagonal=1:size(X,2))

Generalized inverse via sweep operator.
Returns minus the symmetric inverse (negated so callers negate back).
"""
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

"""
    basis!(XX::Symmetric; has_intercept=false)

Returns base of X = [A B C ...]. Takes as input the matrix X'X (actually just its upper-triangular part).
Collinear columns must come first in the base, in the order of columns,
i.e. [A B A] returns [true true false] not [false true true].
"""
function basis!(XX::Symmetric; has_intercept = false)
    invXX = invsym!(XX; has_intercept = has_intercept, setzeros = true)
    return diag(invXX) .< 0
end


"""
    ls_solve!(Xy::Symmetric, nx)

Solve X \\ y. Takes as input the matrix `[X'X X'y; y'X y'y]` (only upper triangle matters).
"""
function ls_solve!(Xy::Symmetric, nx)
    if nx > 0
        invsym!(Xy, diagonal = 1:nx)
        return Xy[1:nx, (nx+1):end]
    else
        return zeros(Float64, 0, size(Xy, 2) - nx)
    end
end

"""
    upper_block_symmetric(A11, A12, A13, A22, A23, A33)
    upper_block_symmetric(A11, A12, A22)

Build a Symmetric matrix from upper-triangular blocks, filling the lower triangle with zeros.
"""
function upper_block_symmetric(A11, A12, A13, A22, A23, A33)
    n1, n2, n3 = size(A11, 1), size(A22, 1), size(A33, 1)
    Symmetric(hvcat(3, A11, A12, A13,
                       zeros(n2, n1), A22, A23,
                       zeros(n3, n1), zeros(n3, n2), A33))
end

function upper_block_symmetric(A11, A12, A22)
    n2 = size(A22, 1)
    Symmetric(hvcat(2, A11, A12,
                       zeros(n2, size(A11, 1)), A22))
end

"""
    collinearity!(Xexo, Xendo, Z, has_intercept, has_iv, coefnames_endo)

Remove collinear variables and, for IV models, build the 2SLS projection.
Returns `(Xexo, Xendo, Z, X, Xhat, XhatXhat, basis_coef, perm, Xendo_res, Z_res, Pi)`.
For non-IV models, `perm`, `Xendo_res`, `Z_res`, and `Pi` are `nothing`.
"""
function collinearity!(
    Xexo::Matrix{Float64},           # exogenous regressors (nobs × kexo)
    Xendo::Matrix{Float64},          # endogenous regressors (nobs × kendo; empty if no IV)
    Z::Matrix{Float64},              # instruments (nobs × kinst; empty if no IV)
    has_intercept::Bool,             # whether the model includes an intercept
    has_iv::Bool,                    # whether the model uses instrumental variables
    coefnames_endo::Vector           # names of endogenous variables (for info messages)
)
    perm = nothing
    if has_iv
        # first pass: remove collinear variables in Xendo
        XendoXendo = Xendo' * Xendo
        basis_endo = basis!(Symmetric(copy(XendoXendo)); has_intercept = false)
        if !all(basis_endo)
            Xendo = Xendo[:, basis_endo]
            XendoXendo = XendoXendo[basis_endo, basis_endo]
        end

        # second pass: remove collinear variables in Xexo, Z, and Xendo
        XexoXexo = Xexo'Xexo
        XexoZ = Xexo'Z
        XexoXendo = Xexo'Xendo
        ZZ = Z'Z
        ZXendo = Z'Xendo
        XexoZXendo = upper_block_symmetric(XexoXexo, XexoZ, XexoXendo, ZZ, ZXendo, XendoXendo)
        basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
        basis_Xexo, basis_Z, basis_endo_small = basis_all[1:size(Xexo, 2)], basis_all[(size(Xexo, 2)+1):(size(Xexo, 2)+size(Z, 2))], basis_all[(size(Xexo, 2)+size(Z, 2)+1):end]

        # if adding Xexo and Z makes Xendo collinear, reclassify as exogenous
        if !all(basis_endo_small)
            Xexo = hcat(Xexo, Xendo[:, .!basis_endo_small])
            Xendo = Xendo[:, basis_endo_small]
            XexoXexo = Xexo'Xexo
            XexoZ = Xexo'Z
            XexoXendo = Xexo'Xendo
            ZXendo = Z'Xendo
            XendoXendo = Xendo'Xendo

            basis_endo2 = trues(length(basis_endo))
            basis_endo2[basis_endo] = basis_endo_small
            endo_reorder = 1:length(basis_endo)
            endo_reorder = vcat(endo_reorder[.!basis_endo2], endo_reorder[basis_endo2])
            perm = vcat(1:length(basis_Xexo), length(basis_Xexo) .+ endo_reorder)
            out = join(coefnames_endo[.!basis_endo2], " ")
            @info "Endogenous vars collinear with ivs. Recategorized as exogenous: $(out)"

            # third pass
            XexoZXendo = upper_block_symmetric(XexoXexo, XexoZ, XexoXendo, ZZ, ZXendo, XendoXendo)
            basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
            basis_Xexo, basis_Z, _ = basis_all[1:size(Xexo, 2)], basis_all[(size(Xexo, 2)+1):(size(Xexo, 2)+size(Z, 2))], basis_all[(size(Xexo, 2)+size(Z, 2)+1):end]
        end
        if !all(basis_Xexo)
            Xexo = Xexo[:, basis_Xexo]
            XexoXexo = XexoXexo[basis_Xexo, basis_Xexo]
            XexoXendo = XexoXendo[basis_Xexo, :]
        end
        if !all(basis_Z)
            Z = Z[:, basis_Z]
            ZZ = ZZ[basis_Z, basis_Z]
            ZXendo = ZXendo[basis_Z, :]
        end
        XexoZ = XexoZ[basis_Xexo, basis_Z]
        size(ZXendo, 1) >= size(ZXendo, 2) || throw(ArgumentError("Model not identified. There must be at least as many ivs as endogeneous variables"))

        basis_endo2 = trues(length(basis_endo))
        basis_endo2[basis_endo] = basis_endo_small
        basis_coef = vcat(basis_Xexo, basis_endo[basis_endo2])

        # Build Xhat via 2SLS
        newZ = hcat(Xexo, Z)
        newZnewZ = hvcat(2, XexoXexo, XexoZ, XexoZ', ZZ)
        newZXendo = vcat(XexoXendo, ZXendo)
        Pi = ls_solve!(Symmetric(hvcat(2, newZnewZ, newZXendo,
                                zeros(size(newZXendo')), zeros(size(Xendo, 2), size(Xendo, 2)))),
                       size(newZnewZ, 2))
        newnewZ = newZ * Pi
        Xhat = hcat(Xexo, newnewZ)
        XhatXhat = upper_block_symmetric(XexoXexo, Xexo'newnewZ, newnewZ'newnewZ)
        X = hcat(Xexo, Xendo)

        # prepare residuals for first stage F statistic
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        Pi2 = ls_solve!(Symmetric(hvcat(2, XexoXexo, XexoZ,
                                zeros(size(Z, 2), size(Xexo, 2)), ZZ)), size(Xexo, 2))
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)

        return Xexo, Xendo, Z, X, Xhat, XhatXhat, basis_coef, perm, Xendo_res, Z_res, Pi
    else
        # get linearly independent columns
        XexoXexo = Xexo'Xexo
        basis_Xexo = basis!(Symmetric(copy(XexoXexo)); has_intercept = has_intercept)
        if !all(basis_Xexo)
            Xexo = Xexo[:, basis_Xexo]
            XexoXexo = XexoXexo[basis_Xexo, basis_Xexo]
        end
        Xhat = Xexo
        XhatXhat = Symmetric(XexoXexo)
        X = Xexo
        basis_coef = basis_Xexo
        return Xexo, Xendo, Z, X, Xhat, XhatXhat, basis_coef, perm, nothing, nothing, nothing
    end
end

"""
    reinsert_omitted!(coef, matrix_vcov, basis_coef, perm)

Expand coefficient vector and vcov matrix to account for omitted (collinear)
variables and IV-reclassified variable permutations.
"""
function reinsert_omitted!(
    coef::Vector{Float64},                   # estimated coefficients (excluding omitted)
    matrix_vcov::Union{Symmetric, Matrix},   # variance-covariance matrix
    basis_coef::BitVector,                   # true for linearly independent columns
    perm::Union{Nothing, Vector{Int}}        # permutation from IV reclassification, or nothing
)
    if !all(basis_coef)
        newcoef = zeros(length(basis_coef))
        newmatrix_vcov = fill(NaN, (length(basis_coef), length(basis_coef)))
        newindex = [searchsortedfirst(cumsum(basis_coef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
            for j in eachindex(newindex)
                newmatrix_vcov[newindex[i], newindex[j]] = matrix_vcov[i, j]
            end
        end
        coef = newcoef
        matrix_vcov = Symmetric(newmatrix_vcov)
    end
    if perm !== nothing
        _invperm = invperm(perm)
        coef = coef[_invperm]
        newmatrix_vcov = zeros(size(matrix_vcov))
        for i in 1:size(newmatrix_vcov, 1)
            for j in 1:size(newmatrix_vcov, 1)
                newmatrix_vcov[i, j] = matrix_vcov[_invperm[i], _invperm[j]]
            end
        end
        matrix_vcov = Symmetric(newmatrix_vcov)
    end
    return coef, matrix_vcov
end
