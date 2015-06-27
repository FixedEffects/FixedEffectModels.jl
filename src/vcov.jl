##############################################################################
##
## AbstractVcovData (and its children) has four important methods: 
## residuals
## regressormatrix
## hatmatrix (by default (X'X)^{-1})
## number of obs
## degree of freedom
##
##############################################################################

abstract AbstractVcovData
residuals(x::AbstractVcovData) = error("not defined")
regressormatrix(x::AbstractVcovData) = error("not defined")
df_residual(x::AbstractVcovData) = size(regressormatrix(x), 2)
nobs(x::AbstractVcovData) = size(regressormatrix(x), 1)
function hatmatrix(x::AbstractVcovData) 
	temp = At_mul_B(regressormatrix(x), regressormatrix(x))
	H = inv(cholfact!(temp))
end


# An implementation of this abstract type that just holds everything
immutable type VcovDataHat <: AbstractVcovData
	regressormatrix::Matrix{Float64} 
	hatmatrix::Matrix{Float64} 
	residuals::Vector{Float64}
	df_residual::Int
end
residuals(x::VcovDataHat) = x.residuals
regressormatrix(x::VcovDataHat) = x.regressormatrix
hatmatrix(x::VcovDataHat) = x.hatmatrix
df_residual(x::VcovDataHat) = x.df_residual

# convert a linear model into VcovDataHat
function VcovDataHat(x::LinearModel) 
	VcovDataHat(x.pp.X, inv(cholfact(x)), residuals(x), size(x.pp.X, 1))
end


##############################################################################
##
## AbstractVcov (and its children) has two methods: 
## allvars that returns variables needed in the dataframe
## vcov!, that returns a covariance matrix. It may change regressormatrix in place, (but not hatmatrix).
##
##############################################################################

abstract AbstractVcov
vcov!(x::AbstractVcovData, v::AbstractVcov) = error("not defined")

# These default methods will be called for errors that do not require access to variables from the initial dataframe (like simple and White standard errors)
allvars(x::AbstractVcov) = nothing
vcov!(x::AbstractVcovData, v::AbstractVcov, df::AbstractDataFrame) = vcov!(x::AbstractVcovData, v::AbstractVcov)

#
# simple standard errors
#

immutable type VcovSimple <: AbstractVcov end

function vcov!(x::AbstractVcovData, t::VcovSimple)
 	scale!(hatmatrix(x), abs2(norm(residuals(x), 2)) /  df_residual(x))
end


#
# White standard errors
#

immutable type VcovWhite <: AbstractVcov end

function vcov!(x::AbstractVcovData, t::VcovWhite) 
	Xu = broadcast!(*,  regressormatrix(x), regressormatrix(x), residuals(x))
	S = At_mul_B(Xu, Xu)
	scale!(S, nobs(x)/df_residual(x))
	sandwich(hatmatrix(x), S) 
end

function sandwich(H::Matrix{Float64}, S::Matrix{Float64})
	H * S * H
end

#
# Clustered standard errors
#

immutable type VcovCluster  <: AbstractVcov
	clusters::Vector{Symbol}
end
VcovCluster(x::Symbol) = VcovCluster([x])

allvars(x::VcovCluster) = x.clusters

function vcov!(x::AbstractVcovData, v::VcovCluster, df::AbstractDataFrame) 
	# Cameron, Gelbach, & Miller (2011).
	df = df[v.clusters]
	X = regressormatrix(x)
	Xu = broadcast!(*,  X, X, residuals(x))
	S = fill(zero(Float64), (size(X, 2), size(X, 2)))
	for i in 1:length(v.clusters)
		for c in combinations(v.clusters, i)
			if length(c) == 1
				# no need to group in this case: it is already a PooledDataArray
				# but there may be less groups then length(f.pool)
				f = df[c[1]]
				fsize = length(unique(f.refs))
			else
				f = group(df[c])
				fsize = length(f.pool)
			end
			if rem(length(c), 2) == 1
				S += helper_cluster(Xu, f, fsize)
			else
				S -= helper_cluster(Xu, f, fsize)
			end
		end
	end
	scale!(S, (nobs(x)-1) / df_residual(x))
	sandwich(hatmatrix(x), S)
end

function helper_cluster(Xu::Matrix{Float64}, f::PooledDataArray, fsize::Int64)
	refs = f.refs
	if fsize == size(Xu, 1)
		# if only one obs by pool, use White, as in Petersen (2009) & Thomson (2011)
		return(At_mul_B(Xu, Xu))
	else
		# otherwise
		X2 = fill(zero(Float64), (fsize, size(Xu, 2)))
		aggregate_matrix!(X2, Xu, refs)
		out = At_mul_B(X2, X2)
		scale!(out, fsize / (fsize- 1))
		return(out)
	end
end

function aggregate_matrix!{T <: Integer}(X2::Matrix{Float64}, Xu::Matrix{Float64}, refs::Vector{T})
	for j in 1:size(Xu, 2)
		 @inbounds @simd for i in 1:size(Xu, 1)
			X2[refs[i], j] += Xu[i, j]
		end
	end
end



#
# HAC standard errors. Not tested.
#

immutable type VcovHac <: AbstractVcov
	time::Symbol
	nlag::Int
	weightfunction::Function
end
# default uses the Bartlett kernel 
VcovHac(time, nlag) = VcovHac(time, nlag, (i, n) -> 1 - i/(n+1))

allvars(x::VcovHac) = x.time

function vcov!(x::AbstractVcovData, v::VcovHac, df)
	time = df[v.time]
	nlag = v.nlag
	weights = map(i -> v.weightfunction(i, nlag), [1:nlag])

	Xu = broadcast!(*,  regressormatrix(x), regressormatrix(x), residuals(x))
	rhos = Array(Matrix{Float64}, nlag)

	# 1 is juste White
	rhos[1] = At_mul_B(Xu, Xu)	

	for i in 2:nlags
		lagx = lag(x, i, time)
		isna = sum(isna(x), 2) + sum(isna(lagx), 2)
		rhos[i] = At_mul_B(x[!isna, :], lagx[!isna, :]) +  At_mul_B(lagx[!isna, :], x[!isna, :]) 
		scale!(rhos[i], weights[i] * (size(x, 1) - i) / (size(x, 1) -i - lenth(isna)))
	end
	S = sum(rhos)
	scale!(S, nobs(x)/df_residual(x))
	sandwich(hatmatrix(x), S) 
end

function lag(x::Array, n::Int, time::Vector) 
	index = findin(time - n, time)
	lagx = DataArray(eltype(x), dim(x))
	lagx[index] = x[index]
	return(lagx)
end



