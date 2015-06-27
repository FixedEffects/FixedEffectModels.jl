
##############################################################################
##
## group transform multiple PooledDataArray into one
## Output is a PooledArray where pool is type Int64, equal to ranking of group
## NA in some row mean result has NA on this row
## 
##############################################################################

function reftype(sz) 
	sz <= typemax(Uint8)  ? Uint8 :
	sz <= typemax(Uint16) ? Uint16 :
	sz <= typemax(Uint32) ? Uint32 :
	Uint64
end

#  similar todropunusedlevels! but (i) may be NA (ii) change pool to integer
function factorize!(refs::Array)
	uu = unique(refs)
	sort!(uu)
	has_na = uu[1] == 0
	T = reftype(length(uu)-has_na)
	dict = Dict(uu, (1-has_na):convert(T, length(uu)-has_na))
	@inbounds @simd for i in 1:length(refs)
		 refs[i] = dict[refs[i]]
	end
	PooledDataArray(RefArray(refs), [1:(length(uu)-has_na);])
end

function pool_combine!{T}(x::Array{Uint64, T}, dv::PooledDataArray, ngroups::Int64)
	@inbounds for i in 1:length(x)
	    # if previous one is NA or this one is NA, set to NA
	    x[i] = (dv.refs[i] == 0 || x[i] == zero(Uint64)) ? zero(Uint64) : x[i] + (dv.refs[i] - 1) * ngroups
	end
	return(x, ngroups * length(dv.pool))
end



function group(x::AbstractVector) 
	v = PooledDataArray(x)
	PooledDataArray(RefArray(v.refs), [1:length(v.pool);])
end
# faster specialization
function group(x::PooledDataArray)
	PooledDataArray(RefArray(copy(x.refs)), [1:length(x.pool);])
end
function group(df::AbstractDataFrame) 
	ncols = size(df, 2)
	v = df[1]
	ncols = size(df, 2)
	ncols == 1 && return(group(v))
	if typeof(v) <: PooledDataArray
		x = convert(Array{Uint64}, v.refs)
	else
		v = PooledDataArray(v, v.na, Uint64)
		x = v.refs
	end
	ngroups = length(v.pool)
	for j = 2:ncols
		v = PooledDataArray(df[j])
		(x, ngroups) = pool_combine!(x, v, ngroups)
	end
	return(factorize!(x))
end
group(df::AbstractDataFrame, cols::Vector) =  group(df[cols])


##############################################################################
##
## sum of square
##
##############################################################################



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



##############################################################################
##
## read formula
##
##############################################################################


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


##############################################################################
##
## build model
##
##############################################################################


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



