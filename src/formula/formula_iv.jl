# decompose formula into normal vs iv part
function is_iv(ex::Expr)
	if (ex.head == :macrocall && ex.args[1] == Symbol("@~")) || (ex.head == :call && ex.args[1] == :(~)) 
		# expression with ~
		length(ex.args) == 3 || error("malformed expression in formula")
		has_iv = true
		endos = ex.args[2]
		ivs = ex.args[3]
	else
		has_iv = false
		endos = nothing
		ivs = nothing
	end
	return has_iv, endos, ivs
end
is_iv(ex) = false, nothing, nothing

function decompose_iv!(rf::Formula)
	iv_formula = nothing
	iv_terms = nothing
	endo_formula = nothing
	endo_terms = nothing
	if !isa(rf.rhs, Expr)
		# symbol or Int 0
		# case with exactly 1 exogenous, 0 endogeneous
		has_iv = false
	else
		has_iv, endos, ivs = is_iv(rf.rhs)
		if has_iv
			# case without exogeneous variables
			has_iv = true
			endo_formula = @eval(@formula($nothing ~ $(endos)))
			iv_formula = @eval(@formula($nothing ~ $(ivs)))
			rf.rhs = :1
		elseif (rf.rhs.head == :call) && (rf.rhs.args[1] == :(+))
			# case with exogeneous variable(s)
			i = 2
			while !has_iv && i <= length(rf.rhs.args)
				has_iv, endos, ivs = is_iv(rf.rhs.args[i])
				if has_iv
					endo_formula = @eval(@formula($nothing ~ $(endos)))
					iv_formula = @eval(@formula($nothing ~  $(ivs)))
					splice!(rf.rhs.args, i)
				else
					i += 1
				end
			end
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
	(has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose_iv!(rf)
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
function nonmissing(mf::ModelFrame)
	if  :msng ∈ fieldnames(typeof(mf))
		mf.msng
	else
		mf.nonmissing
	end
end

# copy ModelFrame but modify to have esample as argument
function ModelFrame2(trms::Terms, d::AbstractDataFrame, esample; contrasts::Dict = Dict())
	df = DataFrame(map(x -> d[x], trms.eterms), Symbol.(string.(trms.eterms)))
	if !all(esample)
		# doing view takes more time and memory for some reason (both in in ModelFrame2 and in evalConstrasts)
		# df = view(df, esample, names(df))
		df = df[esample, :]
	end
	evaledContrasts = evalcontrasts(df, contrasts)
	## Check for non-redundant terms, modifying terms in place
	check_non_redundancy!(trms, df)
	ModelFrame(df, trms, esample, evaledContrasts)
end





# Directly from DataFrames.jl
function dropresponse(trms::Terms)
    if trms.response
        ckeep = 2:size(trms.factors, 2)
        rkeep = vec(any(trms.factors[:, ckeep], dims = 2))
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
allvars(::Any) = Array{Symbol}(undef, 0)




