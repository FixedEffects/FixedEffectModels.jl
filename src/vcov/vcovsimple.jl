type VcovSimple <: AbstractVcovMethod end
type VcovSimpleData <: AbstractVcovMethodData end
VcovMethodData(::VcovSimple, ::AbstractDataFrame) = VcovSimpleData()
function vcov!(::VcovSimpleData, x::VcovData)
    invcrossmatrix = inv(x.crossmatrix)
    scale!(invcrossmatrix, sumabs2(x.residuals) /  x.df_residual)
    return invcrossmatrix
end
shat!(::VcovSimpleData, x::VcovData) = scale(x.crossmatrix, sumabs2(x.residuals))
