# decompose formula into normal vs iv part
function decompose_iv(f::FormulaTerm)
	has_iv = false
	iv_formula = nothing
	iv_terms = nothing
	endo_formula = nothing
	endo_terms = nothing
	rf = f
	for term in eachterm(f.rhs)
		if isa(term, FormulaTerm)
			has_iv = true
			endo_terms = term.lhs
			iv_terms = term.rhs
		end
		rf = FormulaTerm(f.lhs, tuple((t for t in eachterm(f.rhs) if !isa(t, FormulaTerm))...))
	end
	if isempty(rf.rhs)
		rf = FormulaTerm(f.lhs, ConstantTerm(1))
	end
	return rf, has_iv, endo_terms, iv_terms
end


function secondstage(f::FormulaTerm)
	rf, has_iv, endo_terms, iv_terms = decompose_iv(f)
	if has_iv
		rf = FormulaTerm(rf.lhs, (tuple(eachterm(rf.rhs)..., eachterm(endo_terms)...)))
	end
	FormulaTerm(rf.lhs, MatrixTerm(rf.rhs))
end


##############################################################################
##
## build model
##
##############################################################################
function nonmissing(mf::ModelFrame)
	if  :msng ∈ fieldnames(typeof(mf))
		mf.msng
	else
		mf.nonmissing
	end
end


eachterm(x::AbstractTerm) = (x,)
eachterm(x::NTuple{N, AbstractTerm}) where {N} = x


allvars(::Nothing) = Array{Symbol}(undef, 0)
allvars(term::Union{AbstractTerm, NTuple{N, AbstractTerm}}) where {N} = termvars(term)
allvars(f::Union{Expr, Symbol}) = termvars(@eval(@formula(nothing ~$(f))).rhs)

