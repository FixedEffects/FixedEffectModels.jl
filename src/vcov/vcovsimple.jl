VcovFormula(::Type{Val{:simple}}) = VcovSimpleFormula()

struct VcovSimpleFormula <: AbstractVcovFormula end


struct VcovSimpleMethod <: AbstractVcovMethod end
VcovMethod(::AbstractDataFrame, ::VcovSimpleFormula) = VcovSimpleMethod()

function vcov!(::VcovSimpleMethod, x::VcovData)
    invcrossmatrix = Matrix(inv(x.crossmatrix))
    rmul!(invcrossmatrix, sum(abs2, x.residuals) /  x.dof_residual)
    return Symmetric(invcrossmatrix)
end
shat!(::VcovSimpleMethod, x::VcovData) = scale(x.crossmatrix, sumabs2(x.residuals))