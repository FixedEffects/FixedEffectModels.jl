##############################################################################
##
## helper functions
##
##############################################################################

function reftype(sz) 
	sz <= typemax(Uint8)  ? Uint8 :
	sz <= typemax(Uint16) ? Uint16 :
	sz <= typemax(Uint32) ? Uint32 :
	Uint64
end


#  similar todrop unused levels but (i) may be NA (ii) change pool to integer
function factorize!(refs::Array)
	uu = unique(refs)
	vv = setdiff(uu, 0)
	sort!(vv)
	T = reftype(length(vv))
	dict = Dict(vv, 1:convert(T, length(vv)))
	dict[0] = 0
	@inbounds @simd for i in 1:length(refs)
		 refs[i] = dict[refs[i]]
	end
	PooledDataArray(RefArray(refs), [1:length(vv);])
end



function make_integer{T, R, N}(f::PooledDataArray{T, R, N}; skipna = true)
	if skipna
		PooledDataArray(RefArray(f.refs), [1:length(f.pool);])
	else
		index = findfirst(f.refs, 0)
		if index == 0 
			PooledDataArray(RefArray(f.refs), [1:length(f.pool);])
		else
			newvalue = length(f.pool) + 1
			NR = reftype(newvalue)
			refs = convert(Array{NR}, f.refs)
			newvalue = convert(NR, newvalue)
			for i in 1:length(f.refs)
				refs[i] = refs[i] == 0 ? newvalue : refs[i]
			end
			PooledDataArray(RefArray(refs), [1:newvalue;])
		end
	end
end


# group transform multiple vectors into one PooledDataArray
# with skipna = false, NA is max(value) + 1
function group(df::AbstractDataFrame; skipna = true) 
	ncols = size(df, 2)
	dv = PooledDataArray(df[1])
	dv = make_integer(dv; skipna = skipna)
	if ncols == 1
		return(dv)
	else
		x = convert(Vector{Uint64}, dv.refs)
		ngroups = length(dv.pool)
		if skipna
			for j = 2:ncols
				dv = PooledDataArray(df[j])
				for i in 1:size(df, 1)
					# if previous one is NA or this one is NA, set to NA
					x[i] = (dv.refs[i] == 0 || x[i] == zero(Uint64)) ? zero(Uint64) : x[i] + (dv.refs[i] - 1) * ngroups
				end
				ngroups = ngroups * length(dv.pool)
			end
		else
			for j = (ncols - 1):-1:1
				dv = PooledDataArray(df[j])
				for i in 1:size(df, 1)
					x[i] += dv.refs[i] == 0 ? length(dv.pool) * ngroups : (dv.refs[i] - 1) * ngroups
				end
				ngroups = ngroups * (length(dv.pool) + 1)
			end
		end
		return(factorize!(x))
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


#  remove observations with negative weights
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


# used when removing certain rows in a dataset
# NA always removed
function dropUnusedLevels!(f::PooledDataArray)
	uu = unique(f.refs)
	length(uu) == length(f.pool) && return f
	sort!(uu)
	T = reftype(length(uu))
	dict = Dict(uu, 1:convert(T, length(uu)))
	@inbounds @simd  for i in 1:length(f.refs)
		 f.refs[i] = dict[f.refs[i]]
	end
	f.pool = f.pool[uu]
	f
end

dropUnusedLevels!(f::DataArray) = f



