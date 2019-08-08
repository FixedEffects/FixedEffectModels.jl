# decompose formula into normal vs iv part
function decompose_iv(f::FormulaTerm)
	has_iv = false
	iv_formula = nothing
	iv_terms = nothing
	endo_formula = nothing
	endo_terms = nothing
	rf = f
	if isa(f.rhs, FormulaTerm)
		has_iv = true
		endo_terms = f.rhs.lhs
		iv_terms = f.rhs.rhs
		rf = FormulaTerm(f.lhs, ConstantTerm(1))
	elseif isa(f.rhs, Tuple)
		for t in f.rhs
			if isa(t, FormulaTerm)
				has_iv = true
				endo_terms = t.lhs
				iv_terms = t.rhs
			end
		end
		rf = FormulaTerm(f.lhs, tuple((t for t in f.rhs if !isa(t, FormulaTerm))...))
	end
	return rf, has_iv, endo_terms, iv_terms
end


function secondstage(f::FormulaTerm)
	rf, has_iv, endo_terms, iv_terms = f
	if has_iv
		rf = FormulaTerm(rf.lhs, tuple(rf.rhs..., endo_terms...))
	end
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

allvars(f::Expr) = allvars(terms(@eval(@formula(nothing ~$(f))).rhs))
allvars(f::FormulaTerm) = allvars(terms(f.rhs))
allvars(terms::Vector{Term}) = unique([t.sym for t in terms])
allvars(::Any) = Array{Symbol}(undef, 0)





