struct Simple <: AbstractVcov end
simple() = Simple()

struct SimpleMethod <: AbstractVcovMethod end

VcovMethod(::AbstractDataFrame, ::Simple) = SimpleMethod()

function vcov!(::SimpleMethod, x::VcovData)
    invcrossmatrix = Matrix(inv(crossmatrix(x)))
    rmul!(invcrossmatrix, sum(abs2, residuals(x)) /  dof_residual(x))
    return Symmetric(invcrossmatrix)
end
shat!(::SimpleMethod, x::VcovData) = scale(crossmatrix(x), sumabs2(residuals(x)))
