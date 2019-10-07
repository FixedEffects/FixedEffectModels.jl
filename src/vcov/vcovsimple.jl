


struct Simple <: AbstractVcov end
simple() = Simple()

struct SimpleMethod <: AbstractVcovMethod end

VcovMethod(::AbstractDataFrame, ::Simple) = SimpleMethod()

function vcov!(::SimpleMethod, x::VcovData)
    invcrossmatrix = Matrix(inv(x.crossmatrix))
    rmul!(invcrossmatrix, sum(abs2, x.residuals) /  x.dof_residual)
    return Symmetric(invcrossmatrix)
end
shat!(::SimpleMethod, x::VcovData) = scale(x.crossmatrix, sumabs2(x.residuals))
