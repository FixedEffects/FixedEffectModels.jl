# decompose formula into normal vs iv part
function decompose_iv(f::FormulaTerm)
	formula_endo = nothing
	formula_iv = nothing
	for term in eachterm(f.rhs)
		if isa(term, FormulaTerm)
			if formula_endo != nothing
				throw("There can only be one instrumental variable specification")
			end
			formula_endo = FormulaTerm(ConstantTerm(0), tuple(ConstantTerm(0), eachterm(term.lhs)...))
			formula_iv = FormulaTerm(ConstantTerm(0), tuple(ConstantTerm(0), eachterm(term.rhs)...))
		end
	end
	return FormulaTerm(f.lhs, tuple((term for term in eachterm(f.rhs) if !isa(term, FormulaTerm))...)), formula_endo, formula_iv
end




##############################################################################
##
## build model
##
##############################################################################
eachterm(x::AbstractTerm) = (x,)
eachterm(x::NTuple{N, AbstractTerm}) where {N} = x


allvars(::Nothing) = Array{Symbol}(undef, 0)
allvars(term::Union{AbstractTerm, NTuple{N, AbstractTerm}}) where {N} = StatsModels.termvars(term)
allvars(f::Union{Expr, Symbol}) = StatsModels.termvars(@eval(@formula(nothing ~$(f))).rhs)

