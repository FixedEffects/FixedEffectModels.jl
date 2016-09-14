
##############################################################################
##
## read formula
##
##############################################################################

function decompose!(rf::Formula)
	(has_absorb, absorb_formula, absorb_terms) = decompose_absorb!(rf)
	(has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose_iv!(rf)
	return (has_absorb, absorb_formula, absorb_terms, has_iv, iv_formula, iv_terms, endo_formula, endo_terms)
end

# decompose formula into normal + iv vs absorbpart
function decompose_absorb!(rf::Formula)
	has_absorb = false
	absorb_formula = nothing
	absorb_terms = nothing
	if typeof(rf.rhs) == Expr && rf.rhs.args[1] == :(|>)
		has_absorb = true
		absorb_formula = Formula(nothing, rf.rhs.args[3])
		rf.rhs = rf.rhs.args[2]
		absorb_terms = Terms(absorb_formula)
	end
	return(has_absorb, absorb_formula, absorb_terms)
end

# decompose formula into normal vs iv part
function decompose_iv!(rf::Formula)
	has_iv = false
	iv_formula = nothing
	iv_terms = nothing
	endo_formula = nothing
	endo_terms = nothing

	if typeof(rf.rhs) == Expr
		if rf.rhs.head == :(=)
			has_iv = true
			if isa(rf.rhs.args[2], Expr) && rf.rhs.args[2].head == :block
				# happens when several endogeneous variable
				iv_formula = Formula(nothing,  rf.rhs.args[2].args[2])
			else
				iv_formula = Formula(nothing,  rf.rhs.args[2])
			end
			endo_formula = Formula(nothing, rf.rhs.args[1])
			rf.rhs = :1
		elseif rf.rhs.head == :call
			i = 1
			while !has_iv && i <= length(rf.rhs.args)
				if isa(rf.rhs.args[i], Expr) && rf.rhs.args[i].head == :(=)
					has_iv = true
					iv_vars = rf.rhs.args[2]
					if isa(rf.rhs.args[i].args[2], Expr) && rf.rhs.args[i].args[2].head == :block
						# happens when several endogeneous variable
						iv_formula = Formula(nothing,  rf.rhs.args[i].args[2].args[2])
					else
						iv_formula = Formula(nothing,  rf.rhs.args[i].args[2])
					end
					endo_formula = Formula(nothing, rf.rhs.args[i].args[1])
					splice!(rf.rhs.args, i)
				else
					i += 1
				end
			end
		else
			error("formula not correct")
		end
	end

	if has_iv
		iv_terms = Terms(iv_formula)
		iv_terms.intercept = false
		endo_terms = Terms(endo_formula)
		endo_terms.intercept = false
	end
	return(has_iv, iv_formula, iv_terms, endo_formula, endo_terms)
end

function secondstage!(rf::Formula)
	(has_absorb, absorb_formula, absorb_terms, has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose!(rf)
	if has_iv
	    if typeof(rf.rhs) == Symbol
	        rf.rhs = endo_formula.rhs
	    else        
	        push!(rf.rhs.args, endo_formula.rhs)
	    end
	end
end




##############################################################################
##
## build model
##
##############################################################################


function ModelFrame2(trms::Terms, d::AbstractDataFrame, esample; contrasts::Dict = Dict())
	mf = ModelFrame(trms, d; contrasts = contrasts)
	mf.msng = esample
	return mf
end



#  remove observations with negative weights
function isnaorneg{T <: Real}(a::Vector{T}) 
	BitArray(a .> zero(eltype(a)))
end
function isnaorneg{T <: Real}(a::DataVector{T}) 
	out = !a.na
	@simd for i in 1:length(a)
		if out[i]
			@inbounds out[i] = a[i] > zero(Float64)
		end
	end
	BitArray(out)
end

# Directly from DataFrames.jl

function dropresponse(trms::Terms)
    if trms.response
        ckeep = 2:size(trms.factors, 2)
        rkeep = vec(any(trms.factors[:, ckeep], 2))
        Terms(trms.terms, trms.eterms[rkeep], trms.factors[rkeep, ckeep],
              trms.is_non_redundant[rkeep, ckeep], trms.order[ckeep], false, trms.intercept)
    else
        trms
    end
end


function allvars(ex::Expr)
    if ex.head != :call error("Non-call expression encountered") end
    vcat([allvars(a) for a in ex.args[2:end]]...)
end
allvars(f::Formula) = unique(vcat(allvars(f.rhs), allvars(f.lhs)))
allvars(sym::Symbol) = [sym]
allvars(::Any) = Array(Symbol, 0)

# used when removing certain rows in a dataset
# NA always removed
function dropUnusedLevels!(f::PooledDataVector)
	uu = unique(f.refs)
	length(uu) == length(f.pool) && return f
	sort!(uu)
	T = reftype(length(uu))
	dict = Dict{eltype(uu), T}(zip(uu, 1:convert(T, length(uu))))
	@inbounds @simd  for i in 1:length(f.refs)
		 f.refs[i] = dict[f.refs[i]]
	end
	f.pool = f.pool[uu]
	return f
end

dropUnusedLevels!(f::DataVector) = f




