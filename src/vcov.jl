##############################################################################
##
## AbstractVcovData (and its children) has four important methods: residuals, regressors, hatmatrix (by default (X'X)^{-1}), number of obs, degree of freedom
##
##############################################################################

abstract AbstractVcovData
residuals(x::AbstractVcovData) = error("not defined")
regressors(x::AbstractVcovData) = error("not defined")
nobs(x::AbstractVcovData) = size(regressors(x), 1)
df_residual(x::AbstractVcovData) = size(regressors(x), 1)
function hatmatrix(x::AbstractVcovData) 
	temp = At_mul_B(regressors(x), regressors(x))
	H = inv(cholfact!(temp))
end

immutable type VcovData <: AbstractVcovData
	regressors::Matrix{Float64} # If weight, matrix should be X\sqrt{W}
	residuals::Vector{Float64}
	nobs::Int
	df_residual::Int
end
residuals(x::VcovData) = x.residuals
regressors(x::VcovData) = x.regressors
nobs(x::VcovData) = x.nobs
df_residual(x::VcovData) = x.df_residual

immutable type VcovDataHat <: AbstractVcovData
	regressors::Matrix{Float64} # If weight, matrix should be X\sqrt{W}
	hatmatrix::Matrix{Float64} # If weight, matrix should be X\sqrt{W}
	residuals::Vector{Float64}
	nobs::Int
	df_residual::Int
end
residuals(x::VcovDataHat) = x.residuals
regressors(x::VcovDataHat) = x.regressors
hatmatrix(x::VcovDataHat) = x.hatmatrix
nobs(x::VcovDataHat) = x.nobs
df_residual(x::VcovDataHat) = x.df_residual

# convert a linear model into VcovDataHat
function VcovDataHat(x::LinearModel) 
	VcovDataHat(x.pp.X, inv(cholfact(x)), residuals(x), size(x.pp.X, 1), size(x.pp.X, 2))
end


##############################################################################
##
## AbstractVcov (and its children) has two methods: 
## allvars that returns variables needed in the dataframe
## vcov, that returns a covariance matrix
##
##############################################################################


abstract AbstractVcov
allvars(x::AbstractVcov) = nothing
vcov!(x::AbstractVcovData, v::AbstractVcov) = error("not defined")
vcov!(x::AbstractVcovData, v::AbstractVcov, df::AbstractDataFrame) = vcov!(x::AbstractVcovData, v::AbstractVcov)


immutable type VcovSimple <: AbstractVcov 
end

vcov(x::VcovData, t::VcovSimple) = vcov(x)



#
# simple standard errors
#


function vcov!(x::AbstractVcovData, t::VcovSimple)
 	scale!(hatmatrix(x), abs2(norm(residuals(x), 2)) /  df_residual(x))
end


#
# White standard errors
#


immutable type VcovWhite <: AbstractVcov 
end

function vcov!(x::AbstractVcovData, t::VcovWhite) 
	Xu = broadcast!(*,  regressors(x), regressors(x), residuals(x))
	S = At_mul_B(Xu, Xu)
	scale!(S, nobs(x)/df_residual(x))
	sandwich(x, S) 
end


function sandwich(x::AbstractVcovData, S::Matrix{Float64})
	H = hatmatrix(x)
	H * S * H
end


#
# HAC standard errors
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

	Xu = broadcast!(*,  regressors(x), regressors(x), residuals(x))
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
	sandwich(x, S) 
end

function lag(x::Array, n::Int, time::Vector) 
	index = findin(time - n, time)
	lagx = DataArray(eltype(x), dim(x))
	lagx[index] = x[index]
	return(lagx)
end

#
# Cluster standard errors
#

immutable type VcovCluster  <: AbstractVcov
	clusters::Vector{Symbol}
end
VcovCluster(x::Symbol) = VcovCluster([x])

allvars(x::VcovCluster) = x.clusters

# Cameron, Gelbach, & Miller (2011).
function vcov!(x::AbstractVcovData, v::VcovCluster, df::AbstractDataFrame) 
	df = df[v.clusters]
	X = regressors(x)
	Xu = broadcast!(*,  X, X, residuals(x))
	S = fill(zero(Float64), (size(X, 2), size(X, 2)))
	for i in 1:length(v.clusters)
		for c in combinations(v.clusters, i)
			if length(c) == 1
				# because twice faster
				f = df[c[1]]
			else
				f = group(df[c])
			end
			if rem(length(c), 2) == 1
				S += helper_cluster(Xu, f)
			else
				S -= helper_cluster(Xu, f)
			end
		end
	end
	scale!(S, (nobs(x)-1) / df_residual(x))
	sandwich(x, S)
end

function helper_cluster(Xu::Matrix{Float64}, f::PooledDataArray)
	pool = f.pool
	refs = f.refs
	# if only one obs by pool, use White, as in Petersen (2009) & Thomson (2011)
	if length(pool) == size(Xu, 1)
		At_mul_B(Xu, Xu)
		return(At_mul_B(Xu, Xu))
	else
		# otherwise
		X2 = fill(zero(Float64), (length(f.pool), size(Xu, 2)))
		fsize = fill(zero(Int64),  length(f.pool))
		aggregate_matrix!(X2, Xu, refs, fsize)
		out = At_mul_B(X2, X2)
		scale!(out, sum(fsize .> 0) / (sum(fsize .> 0) - 1))
		return(out)
	end
end


function aggregate_matrix!{T <: Integer}(X2::Matrix{Float64}, Xu::Matrix{Float64}, refs::Vector{T}, fsize::Vector{Int64})
	for j in 1:size(Xu, 2)
		if j == 1
			 @inbounds @simd for i in 1:size(Xu, 1)
			 	fsize[refs[i]] += 1
				X2[refs[i], j] += Xu[i, j]
			end
		else
			 @inbounds @simd for i in 1:size(Xu, 1)
				X2[refs[i], j] += Xu[i, j]
			end
		end
	end
end







