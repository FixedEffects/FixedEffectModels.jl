
##############################################################################
##
## getfe computes fixed effect estimates
## b is (y - x'b) - (\overline{y} - \overline{x}'b)
###############################################################################


# Return vector of vector of estimates
function getfe(fixedeffects::Vector{AbstractFixedEffect}, 
			   b::Vector{Float64}; 
			   maxiter = 10_000_000)
	## initialize data structures
	(fevalues, where, refs, A, interceptindex) = initialize(fixedeffects)
	
	# solve Ax = b by kaczmarz algorithm
	kaczmarz!(fevalues, b, refs, A, maxiter, interceptindex)

	# rescale fixed effects 
	if length(interceptindex) >= 2
		components = connectedcomponent(refs, where, interceptindex)
		rescale!(fevalues, components, interceptindex)
	end

	return fevalues
end


# Return dataframe of estimates
function getfe(fixedeffects::Vector{AbstractFixedEffect},
			   b::Vector{Float64}, 
			   esample::BitVector ; 
			   maxiter = 100_000)
	
	# return vector of vector of estimates
	fevalues = getfe(fixedeffects, b, maxiter = maxiter)

	# insert matrix into dataframe
	newdf = DataFrame()
	len = length(esample)
	for j in 1:length(fixedeffects)
		name = fixedeffects[j].id
		T = eltype(fixedeffects[j].refs)
		refs = fill(zero(T), len)
		refs[esample] = fixedeffects[j].refs
		newdf[fixedeffects[j].id] = PooledDataArray(RefArray(refs), fevalues[j])
	end
	return newdf
end




##############################################################################
##
## Initialize structures
## Denote refs(j) the reference vector of the jth factor
## where[j][i] is a set that stores indices k with refs(j)[k] = i
## fevalues[j][i] is a Float64 that stores the fixed effect estimate with refs = i
## refs[j, i] is a refs(j)[i]
##############################################################################

function initialize(fixedeffects::Vector{AbstractFixedEffect})
	nobs = length(fixedeffects[1].refs)
	fevalues = Array(Vector{Float64}, length(fixedeffects)) 
	where = Array(Vector{Set{Int}}, length(fixedeffects))
	refs = fill(zero(Int), length(fixedeffects), nobs)
	A = fill(one(Float64), length(fixedeffects),  nobs)
	interceptindex = Int[]
	j = 0
	for f in fixedeffects
		j += 1
		initialize!(j, f, fevalues, where, refs, A)
		if typeof(f) <: FixedEffectIntercept
			push!(interceptindex, j)
		end
	end
	return (fevalues, where, refs, A, interceptindex)
end

function initialize!(
	j::Int, 
	f::FixedEffectIntercept,
	fevalues::Vector{Vector{Float64}}, 
	where::Vector{Vector{Set{Int}}},
	refs::Matrix{Int},
	A::Matrix{Float64}
 	)
	fevalues[j] = fill(zero(Float64), length(f.scale))
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
	fevalues::Vector{Vector{Float64}}, 
	where::Vector{Vector{Set{Int}}},
	refs::Matrix{Int},
	A::Matrix{Float64}
 	)
	fevalues[j] = fill(zero(Float64), length(f.scale))
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
## Find Fixed Effect : Randomized Kaczmarz algorithm = stochastic gradient descent.
## https://en.wikipedia.org/wiki/Kaczmarz_method
##############################################################################


function kaczmarz!(fevalues::Vector{Vector{Float64}},
				   b::Vector{Float64},
				   refs::Matrix{Int},
				   A::Matrix{Float64},
				   maxiter::Integer,
				   interceptindex::Vector{Int})
	# precompute norm[i] = sum_j A[j, i]^2 for probability distribution
	# precompute invnorm = 1/norm[i] because division costly
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

	# sampling distribution 
	if length(interceptindex) == length(fevalues)
		# if all categorical variables are intercept, uniform sampling
		dist = 1:length(b)
	else
		# otherwise, sample with probability ||a[i]^2||
		# Strohmer and Vershynin (2009): A randomized Kaczmarz algorithm with exponential convergence
		# does it really accelerate the convergence?
		dist = AliasTable(norm/sum(norm))
	end

	kaczmarz!(fevalues, b, refs, A, maxiter, dist, invnorm)
	return fevalues
end

function kaczmarz!(fevalues::Vector{Vector{Float64}},
				   b::Vector{Float64},
			       refs::Matrix{Int},
			       A::Matrix{Float64},
			       maxiter::Integer,
 			       dist,
			       invnorm::Vector{Float64})
	len_fe = length(fevalues)
	len_b = length(b)
	iter = 0
	while iter < maxiter
		iter += 1
		error = zero(Float64)
		@inbounds for k in 1:len_b
			i = rand(dist)
			# get the scale
			numerator = b[i]
			for j in 1:len_fe
				aij = A[j, i]
				numerator -= fevalues[j][refs[j, i]] * aij
			end
			update = numerator * invnorm[i]
			# update
			for j in 1:len_fe	
				change = update * A[j, i]
				fevalues[j][refs[j, i]] += change
				error += abs2(change)
			end
		end
		if error < 1e-15 * len_b
			break
		end
	end
end



##############################################################################
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

function connectedcomponent(refs::Matrix{Int},
							where::Vector{Vector{Set{Int}}},
						    interceptindex::Vector{Int})
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
function connectedcomponent!(component::Vector{Set{Int}},
							 visited::Vector{Bool},
						     i::Integer,
						     refs::Matrix{Int},
						     where::Vector{Vector{Set{Int}}},
						     interceptindex::Vector{Int}) 
	visited[i] = true
	tovisit = Set{Int}()
	for j in interceptindex
		ref = refs[j, i]
		if !(ref in component[j])
			# add node in component
			push!(component[j], ref)
			# add children in list to visit
			for i in where[j][ref]
				push!(tovisit, i)
			end
		end
	end
	for j in tovisit
		connectedcomponent!(component, visited, j, refs, where, interceptindex)
	end
end


##############################################################################
##
## rescale fixed effect to make solution unique
## normalization: for each factor except the first one, mean within each component is 0 
## Unique solution with two components, not really with more
##
###############################################################################

function rescale!(fevalues::Vector{Vector{Float64}}, 
			      components::Vector{Vector{Set{Int}}}, 
			      interceptindex)
	i1 = interceptindex[1]
	adj1 = zero(Float64)
	for component in components
		for i in reverse(interceptindex)
			# demean all fixed effects except the first
			if i != i1
				adji = zero(Float64)
				for j in component[i]
					adji += fevalues[i][j]
				end
				adji /= length(component[i])
				for j in component[i]
					fevalues[i][j] -= adji
				end
				adj1 += adji
			else
				# rescale the first fixed effects
				for j in component[i1]
					fevalues[i1][j] += adj1
				end
			end
		end
	end
end



