VcovFormula(::Type{Val{:simple}}) = VcovSimpleFormula()

mutable struct VcovSimpleFormula <: AbstractVcovFormula end
allvars(x::VcovSimpleFormula) = Symbol[]


mutable struct VcovSimpleMethod <: AbstractVcovMethod end
VcovMethod(::AbstractDataFrame, ::VcovSimpleFormula) = VcovSimpleMethod()

function vcov!(::VcovSimpleMethod, x::VcovData)
    invcrossmatrix = inv(x.crossmatrix)
    scale!(invcrossmatrix, sum(abs2, x.residuals) /  x.df_residual)
    return invcrossmatrix
end
shat!(::VcovSimpleMethod, x::VcovData) = scale(x.crossmatrix, sumabs2(x.residuals))