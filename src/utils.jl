##############################################################################
##
## Fe and Feinteracted
##
##############################################################################

# For each fixed effect, this stores the reference vector (ie a map of each row to a group), the weights, the size of each group, and, for FeInteracted, the interaction variable

abstract AbstractFe

immutable type Fe{R} <: AbstractFe
	refs::Vector{R}
	w::Vector{Float64}
	scale::Vector{Float64}
	name::Symbol
end

immutable type FeInteracted{R} <: AbstractFe
	refs::Vector{R}
	w::Vector{Float64}
	scale::Vector{Float64}
	x::Vector{Float64} # the continuous interaction 
	name::Symbol
	xname::Symbol
end


function Fe(f::PooledDataArray, w::Vector{Float64}, name::Symbol)
	scale = fill(zero(Float64), length(f.pool))
	refs = f.refs
	@inbounds @simd  for i in 1:length(refs)
		 scale[refs[i]] += abs2(w[i])
	end
	@inbounds @simd  for i in 1:length(scale)
		 scale[i] = scale[i] > 0 ? (one(Float64) / scale[i]) : zero(Float64)
	end
	Fe(refs, w, scale, name)
end

function FeInteracted(f::PooledDataArray, w::Vector{Float64}, x::Vector{Float64}, name::Symbol, xname::Symbol)
	scale = fill(zero(Float64), length(f.pool))
	refs = f.refs
	@inbounds @simd  for i in 1:length(refs)
		 scale[refs[i]] += abs2((x[i] * w[i]))
	end
	@inbounds @simd  for i in 1:length(scale)
		scale[i] = scale[i] > 0 ? (one(Float64) / scale[i]) : zero(Float64)
	end
	FeInteracted(refs, w, scale, x, name, xname)
end


function construct_fe(df::AbstractDataFrame, a::Expr, w::Vector{Float64})
	if a.args[1] == :&
		if (typeof(df[a.args[2]]) <: PooledDataArray) & !(typeof(df[a.args[3]]) <: PooledDataArray)
			f = df[a.args[2]]
			x = convert(Vector{Float64}, df[a.args[3]])
			return(FeInteracted(f, w, x, a.args[2], a.args[3]))
		elseif (typeof(df[a.args[3]]) <: PooledDataArray) & !(typeof(df[a.args[2]]) <: PooledDataArray)
			f = df[a.args[3]]
			x = convert(Vector{Float64}, df[a.args[2]])
			return(FeInteracted(f, w, x, a.args[3], a.args[2]))
		else
			error("& is not of the form factor & nonfactor")
		end
	else
		error("Formula should be composed of & and symbols")
	end
end

function construct_fe(df::AbstractDataFrame, a::Symbol, w::Vector{Float64})
	if typeof(df[a]) <: PooledDataArray
		return(Fe(df[a], w, a))
	else
		error("$(a) is not a pooled data array")
	end
end

function construct_fe(df::AbstractDataFrame, v::Vector, w::Vector{Float64})
	factors = AbstractFe[]
	for a in v
		push!(factors, construct_fe(df, a, w))
	end
	return(factors)
end




##############################################################################
##
## Demean algorithm
##
##############################################################################

# Algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf

function demean_vector_factor!(ans::Vector{Float64}, fe::Fe, mean::Vector{Float64})
	scale = fe.scale
	refs = fe.refs
	w = fe.w
	@inbounds @simd  for i in 1:length(ans)
		 mean[refs[i]] += ans[i] * w[i]
	end
	@inbounds @simd  for i in 1:length(scale)
		 mean[i] *= scale[i] 
	end
	@inbounds @simd  for i in 1:length(ans)
		 ans[i] -= mean[refs[i]] * w[i]
	end
	return(ans)
end

function demean_vector_factor!(ans::Vector{Float64}, fe::FeInteracted, mean::Vector{Float64})
	scale = fe.scale
	refs = fe.refs
	x = fe.x
	w = fe.w
	@inbounds @simd  for i in 1:length(ans)
		 mean[refs[i]] += ans[i] * x[i] * w[i]
	end
	@inbounds @simd  for i in 1:length(scale)
		 mean[i] *= scale[i] 
	end
	@inbounds @simd  for i in 1:length(ans)
		 ans[i] -= mean[refs[i]] * x[i] * w[i]
	end
	return(ans)
end

function demean_vector!(x::Vector{Float64}, fes::Vector{AbstractFe})
	tolerance = ((1e-8 * length(x))^2)::Float64
	delta = 1.0
	if length(fes) == 1 && typeof(fes[1]) <: Fe
		max_iter = 1
	else
		max_iter = 1000
	end
	olx = similar(x)
	# allocate array of means for each factor
	dict1 = Dict{AbstractFe, Vector{Float64}}()
	dict2 = Dict{AbstractFe, Vector{Float64}}()
	for fe in fes
		dict1[fe] = zeros(Float64, length(fe.scale))
	end
	for iter in 1:max_iter
		@inbounds @simd  for i in 1:length(x)
			olx[i] = x[i]
		end
		for fe in fes
			mean = dict1[fe]
			fill!(mean, zero(Float64))
			demean_vector_factor!(x, fe, mean)
		end
		delta = sqeuclidean(x, olx)
		if delta < tolerance
			break
		end
	end
	return(x)
end

function demean_vector!(x::DataVector{Float64}, fes::Vector{AbstractFe})
	demean_vector(convert(Vector{Float64}, x), fes)
end



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
	dv = DataArrays.PooledDataArray(df[ncols])
	if skipna
		x = map(z -> convert(Uint64, z), dv.refs)
		ngroups = length(dv.pool)
		for j = (ncols - 1):-1:1
			dv = DataArrays.PooledDataArray(df[j])
			for i = 1:DataFrames.size(df, 1)
				x[i] += ((dv.refs[i] == 0 | x[i] == 0) ? 0 : (dv.refs[i] - 1) * ngroups)
			end
			ngroups = ngroups * length(dv.pool)
		end
		# factorize
		uu = unique(x)
		T = eltype(x)
		vv = setdiff(uu, zero(T))
		dict = Dict(vv, 1:(length(vv)))
		compact(PooledDataArray(DataArrays.RefArray(map(z -> z == 0 ? zero(T) : dict[z], x)),  [1:length(vv);]))
	else
		# code from groupby
		dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
		x = map(z -> convert(Uint64, z) + dv_has_nas, dv.refs)
		ngroups = length(dv.pool) + dv_has_nas
		for j = (ncols - 1):-1:1
			dv = DataArrays.PooledDataArray(df[j])
			dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
			for i = 1:DataFrames.size(df, 1)
				x[i] += (dv.refs[i] + dv_has_nas- 1) * ngroups
			end
			ngroups = ngroups * (length(dv.pool) + dv_has_nas)
		end
		# end of code from groupby
		# factorize
		uu = unique(x)
		T = eltype(x)
		dict = Dict(uu, 1:length(uu))
		compact(PooledDataArray(DataArrays.RefArray(map(z -> dict[z], x)),  [1:length(uu);]))
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



function remove_negweight!{R}(esample::BitArray{1}, w::DataVector{R})
	@inbounds @simd  for (i in 1:length(w))
		 esample[i] = esample[i] && (w[i] > zero(R))
	end
	esample
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
