# decompose formula into normal vs iv part
function decompose_iv(f::FormulaTerm)
	formula = f
	formula_endo = nothing
	formula_iv = nothing
	for term in eachterm(f.rhs)
		if isa(term, FormulaTerm)
			formula_endo = FormulaTerm(ConstantTerm(0), term.lhs)
			formula_iv = FormulaTerm(ConstantTerm(0), term.rhs)
		end
		formula = FormulaTerm(f.lhs, tuple((t for t in eachterm(f.rhs) if !isa(t, FormulaTerm))...))
	end
	return formula, formula_endo, formula_iv
end




##############################################################################
##
## build model
##
##############################################################################
eachterm(x::AbstractTerm) = (x,)
eachterm(x::NTuple{N, AbstractTerm}) where {N} = x


allvars(::Nothing) = Array{Symbol}(undef, 0)
allvars(term::Union{AbstractTerm, NTuple{N, AbstractTerm}}) where {N} = termvars(term)
allvars(f::Union{Expr, Symbol}) = termvars(@eval(@formula(nothing ~$(f))).rhs)

