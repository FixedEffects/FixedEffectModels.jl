
##############################################################################
##
## Fe and FixedEffectSlope
##
###############################################################################


# Starts with X, fixed effect and residuals R
function getfe(fes::Vector{AbstractFixedEffect}, b::Vector{Float64}; maxiter = 10_000_000)
	## initialize data structures
	(fe, where, refs, A, interceptindex) = initialize(fes)
	
	# solve Ax = b by kaczmarz algorithm
	kaczmarz!(fe, b, refs, A, maxiter)

	# rescale fixed effects 
	# rule: mean within each component is 0 (except for the first one)
	if length(interceptindex) >= 2
		components = connectedcomponent(refs, where, interceptindex)
		rescale!(fe, components, interceptindex)
	end

	return fe
end


##############################################################################
##
## Initialize
## 
##############################################################################

function initialize(fes::Vector{AbstractFixedEffect})
	nobs = length(fes[1].refs)
	fe = Array(Vector{Float64}, length(fes)) 
	where = Array(Vector{Set{Int}}, length(fes))
	refs = fill(zero(Int), length(fes), nobs)
	A = fill(one(Float64), length(fes),  nobs)
	interceptindex = Int[]
	j = 0
	for f in fes
		j += 1
		initialize!(j, f, fe, where, refs, A)
		if typeof(f) <: FixedEffectIntercept
			push!(interceptindex, j)
		end
	end
	return (fe, where, refs, A, interceptindex)
end

function initialize!(
	j::Int, 
	f::FixedEffectIntercept,
	fe::Vector{Vector{Float64}}, 
	where::Vector{Vector{Set{Int}}},
	refs::Matrix{Int},
	A::Matrix{Float64}
 	)
	fe[j] = fill(zero(Float64), length(f.scale))
	where[j] = Array(Set{Int}, length(f.scale))
	# fill would create a reference to the same object
	for i in 1:length(f.scale)
		where[j][i] = Set{Int}()
	end
	for i in 1:length(f.refs)
		refi = f.refs[i]
		refs[j, i] = refi
		push!(where[j][refi], i)
	end
end

function initialize!(
	j::Int, 
	f::FixedEffectSlope,
	fe::Vector{Vector{Float64}}, 
	where::Vector{Vector{Set{Int}}},
	refs::Matrix{Int},
	A::Matrix{Float64}
 	)
	fe[j] = fill(zero(Float64), length(f.scale))
	where[j] = Array(Set{Int}, length(f.scale))
	# fill would create a reference to the same object
	for i in 1:length(f.scale)
		where[j][i] = Set{Int}()
	end
	@inbounds for i in 1:length(f.refs)
		refi = f.refs[i]
		refs[j, i] = refi
		A[j, i] = f.interaction[i]
		push!(where[j][refi], i)
	end
end

##############################################################################
##
## Find Fixed Effect : Kaczmarz algorithm
## https://en.wikipedia.org/wiki/Kaczmarz_method
##
##############################################################################


function kaczmarz!(fe::Vector{Vector{Float64}}, b::Vector{Float64}, refs::Matrix{Int}, A::Matrix{Float64}, maxiter::Integer)
	len = length(fe)
	len_b = length(b)
	iter = 0
	x = fill(zero(Int), len_b)

	# precompute norm for probability distribution
	# precompute invonrm because of 1/norm
	norm = fill(zero(Float64), size(A, 2))
	invnorm = fill(zero(Float64), size(A, 2))
	@inbounds for i in 1:size(A, 2)
		out = zero(Float64)
		for j in 1:size(A, 1)
			out += abs2(A[j, i])
		end
		norm[i] = out
		invnorm[i] = 1 / out
	end
	dist = AliasTable(norm/sum(norm))
	while iter < maxiter
		iter += 1
		permutation = rand!(dist, x)
		error = update!(fe, b, refs, A, permutation, invnorm, len)
		if error < 1e-15 * len_b
			break
		end
	end
	return(fe)
end

function update!(fe::Vector{Vector{Float64}}, b::Vector{Float64}, refs::Matrix{Int}, A::Matrix{Float64}, permutation::Vector{Int64}, invnorm::Vector{Float64}, len::Int)
	error = zero(Float64)
	@inbounds for i in permutation
		numerator = b[i]
		denominator = zero(Float64)
		for j in 1:len
			aij = A[j, i]
			numerator -= fe[j][refs[j, i]] * aij
		end
		update = numerator * invnorm[i]
		for j in 1:len		
			change = update * A[j, i]
			fe[j][refs[j, i]] += change
			error += abs2(change)
		end
	end
	return error
end



##############################################################################
##
## Connected component : Breadth-first search
## components is an array of component (length is number of components)
## A component is an array of set (length is number of fe)
##
##############################################################################

function connectedcomponent(refs::Matrix{Int}, where::Vector{Vector{Set{Int}}}, interceptindex::Vector{Int})
	nobs = size(refs, 2)
	visited = fill(false, nobs)
	components = Vector{Set{Int}}[]
	for i in 1:nobs
		if !visited[i]
			component = Array(Set{Int}, length(interceptindex))
			for i in 1:length(interceptindex)
				component[i] = Set{Int}()
			end
			connectedcomponent!(component, visited, i, refs, where, interceptindex)
			push!(components, component)
		end
	end
	return components
end

# Breadth-first search
function connectedcomponent!(component::Vector{Set{Int}}, visited::Vector{Bool}, i::Integer, refs::Matrix{Int}, where::Vector{Vector{Set{Int}}}, interceptindex::Vector{Int}) 
	visited[i] = true
	tovisit = Set{Int}()
	for j in interceptindex
		ref = refs[j, i]
		if !(ref in component[j])
			push!(component[j], ref)
			update!(tovisit, where[j][ref])
		end
	end
	for j in tovisit
		connectedcomponent!(component, visited, j, refs, where, interceptindex)
	end
end


function update!(tovisit::Set{Int}, neighbors::Set{Int})
	for i in neighbors
		if !in(i, tovisit)
			push!(tovisit, i)
		end
	end
end

##############################################################################
##
## rescale fixed effect to make solution unique
##
###############################################################################

function rescale!(fe::Vector{Vector{Float64}}, components::Vector{Vector{Set{Int}}}, interceptindex)
	i1 = interceptindex[1]
	adj1 = zero(Float64)
	for component in components
		for i in reverse(interceptindex)
			# demean all fixed effects except the first
			if i != i1
				adji = zero(Float64)
				for j in component[i]
					adji += fe[i][j]
				end
				adji /= length(component[i])
				for j in component[i]
					fe[i][j] -= adji
				end
				adj1 += adji
			else
				# rescale the first fixed effects
				for j in component[i1]
					fe[i1][j] += adj1
				end
			end
		end
	end
end


##############################################################################
##
## DataFrame version
##
###############################################################################

function getfe(x::Union(RegressionResultFE, RegressionResultFEIV))
	error("Add the original dataframe as a second argument")
end



function getfe(fes::Vector{AbstractFixedEffect}, b::Vector{Float64}, esample::BitVector ; maxiter = 100_000)
	DataFrame(fes, getfe(fes, b, maxiter = maxiter), esample)
end


function DataFrame(fes::Vector{AbstractFixedEffect}, fe::Vector{Vector{Float64}}, esample::BitVector)
	newdf = DataFrame()
	len = length(esample)
	for j in 1:length(fes)
		name = fes[j].id
		T = eltype(fes[j].refs)
		refs = fill(zero(T), len)
		refs[esample] = fes[j].refs
		newdf[fes[j].id] = PooledDataArray(RefArray(refs), fe[j])
	end
	return newdf
end



