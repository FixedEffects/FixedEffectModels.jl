##############################################################################
##
## helper functions
##
##############################################################################

function dropUnusedLevels!(f::PooledDataArray)
	rr = f.refs
	uu = unique(rr)
	f.pool = uu
	T = eltype(rr)
	dict = Dict(uu, 1:convert(Uint32, length(uu)))
	@inbounds @simd  for i in 1:length(rr)
		 rr[i] = dict[rr[i]]
	end
	f
end
dropUnusedLevels!(f::DataArray) = f


# group transform multiple vectors into one PooledDataArray
function group(df::AbstractDataFrame; skipna = true) 
	ncols = length(df)
	dv = PooledDataArray(df[ncols])
	if skipna
		x = map(z -> convert(Uint64, z), dv.refs)
		ngroups = length(dv.pool)
		for j = (ncols - 1):-1:1
			dv = PooledDataArray(df[j])
			for i = 1:size(df, 1)
				x[i] += ((dv.refs[i] == 0 | x[i] == 0) ? 0 : (dv.refs[i] - 1) * ngroups)
			end
			ngroups = ngroups * length(dv.pool)
		end
		# factorize
		uu = unique(x)
		T = eltype(x)
		vv = setdiff(uu, zero(T))
		dict = Dict(vv, 1:(length(vv)))
		compact(PooledDataArray(RefArray(map(z -> z == 0 ? zero(T) : dict[z], x)),  [1:length(vv);]))
	else
		# code from groupby
		dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
		x = map(z -> convert(Uint64, z) + dv_has_nas, dv.refs)
		ngroups = length(dv.pool) + dv_has_nas
		for j = (ncols - 1):-1:1
			dv = PooledDataArray(df[j])
			dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
			for i = 1:size(df, 1)
				x[i] += (dv.refs[i] + dv_has_nas- 1) * ngroups
			end
			ngroups = ngroups * (length(dv.pool) + dv_has_nas)
		end
		# end of code from groupby
		# factorize
		uu = unique(x)
		T = eltype(x)
		dict = Dict(uu, 1:length(uu))
		compact(PooledDataArray(RefArray(map(z -> dict[z], x)),  [1:length(uu);]))
	end
end
group(df::AbstractDataFrame, cols::Vector; skipna = true) =  group(df[cols]; skipna = skipna)






# compute rss and tss from vector of residual and response vector
function compute_ss(residuals::Vector{Float64}, y::Vector{Float64}, hasintercept::Bool)
	ess = abs2(norm(residuals))
	if hasintercept
		tss = zero(Float64)
		m = mean(y)::Float64
		@inbounds @simd  for i in 1:length(y)
			tss += abs2((y[i] - m))
		end
	else
		tss = abs2(norm(y))
	end
	(ess, tss)
end
function compute_ss(residuals::Vector{Float64}, y::Vector{Float64}, hasintercept::Bool, w::Vector{Float64}, sqrtw::Vector{Float64})
	ess = abs2(norm(residuals))
	if hasintercept
		m = (mean(y) / sum(sqrtw) * length(residuals))::Float64
		tss = zero(Float64)
		@inbounds @simd  for i in 1:length(y)
		 tss += abs2(y[i] - sqrtw[i] * m)
		end
	else
		tss = abs2(norm(y))
	end
	(ess, tss)
end



function multiplication_elementwise!(y::Vector{Float64}, sqrtw::Vector{Float64})
	@inbounds @simd  for i in 1:length(y)
		 y[i] *= sqrtw[i] 
	end
	return(y)
end



# decompose formula into normal + iv vs absorbpart
function decompose_absorb!(rf::Formula)
	has_absorb = false
	absorb_vars = nothing
	absorbt = nothing
	if typeof(rf.rhs) == Expr && rf.rhs.args[1] == :(|>)
		has_absorb = true
		absorbf = Formula(nothing, rf.rhs.args[3])
		absorb_vars = unique(allvars(rf.rhs.args[3]))
		absorbt = Terms(absorbf)
		rf.rhs = rf.rhs.args[2]
	end
	return(rf, has_absorb, absorb_vars, absorbt)
end

# decompose formula into normal vs iv part
function decompose_iv!(rf::Formula)
	has_iv = false
	iv_vars = nothing
	ivt = nothing
	if typeof(rf.rhs) == Expr
		if rf.rhs.head == :(=)
			has_iv = true
			iv_vars = unique(allvars(rf.rhs.args[2]))
			ivf = deepcopy(rf)
			ivf.rhs = rf.rhs.args[2]
			ivt = Terms(ivf)
			rf.rhs = rf.rhs.args[1]
		else
			for i in 1:length(rf.rhs.args)
				if typeof(rf.rhs.args[i]) == Expr && rf.rhs.args[i].head == :(=)
					has_iv = true
					iv_vars = unique(allvars(rf.rhs.args[i].args[2]))
					ivf = deepcopy(rf)
					ivf.rhs.args[i] = rf.rhs.args[i].args[2]
					ivt = Terms(ivf)
					rf.rhs.args[i] = rf.rhs.args[i].args[1]
				end
			end
		end
	end
	return(rf, has_iv, iv_vars, ivt)
end

function simpleModelFrame(df, t, esample)
	df1 = DataFrame(map(x -> df[x], t.eterms))
	names!(df1, convert(Vector{Symbol}, map(string, t.eterms)))
	mf = ModelFrame(df1, t, esample)
end


function isnaorneg{T <: Real}(a::Vector{T}) 
	bitpack(a .> zero(eltype(a)))
end

function isnaorneg{T <: Real}(a::DataVector{T}) 
	out = !a.na
	@simd for i in 1:length(a)
		if out[i]
			@inbounds out[i] = a[i] > zero(Float64)
		end
	end
	bitpack(out)
end


function isna2{T <: Real}(a::DataVector{T}) 
	map(x -> isa(x, NAtype), a)
end

# Directly from DataFrames.jl
function remove_response(t::Terms)
    # shallow copy original terms
    t = Terms(t.terms, t.eterms, t.factors, t.order, t.response, t.intercept)
    if t.response
        t.order = t.order[2:end]
        t.eterms = t.eterms[2:end]
        t.factors = t.factors[2:end, 2:end]
        t.response = false
    end
    return t
end


function allvars(ex::Expr)
    if ex.head != :call error("Non-call expression encountered") end
    [[allvars(a) for a in ex.args[2:end]]...]
end
allvars(f::Formula) = unique(vcat(allvars(f.rhs), allvars(f.lhs)))
allvars(sym::Symbol) = [sym]
allvars(v::Any) = Array(Symbol, 0)
