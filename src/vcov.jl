using GLM, DataFrames


abstract AbstractVceModel

abstract AbstractVce
allvars2(x::AbstractVce) = nothing


immutable type VceModel{T} <: AbstractVceModel
	X::Matrix{T} # If weight, matrix should be X\sqrt{W}
	residuals::Vector{T}
	df_residual::Integer
	nobs::Integer
end
VceModel{T}(X::Matrix{T}, residual::Vector{T}) = VceModel(X, residual, size(X, 1) - size(X,2), size(X, 1))

immutable type VceModelH{T} <: AbstractVceModel
	X::Matrix{T} # If weight, matrix should be X\sqrt{W}
	H::Matrix{T} # If weight, matrix should be X\sqrt{W}
	residuals::Vector{T}
	df_residual::Integer
	nobs::Integer
end
VceModelH{T}(X::Matrix{T}, H::Matrix{T}, residual::Vector{T}) = VceModelH(X, H, residual, size(X, 1) - size(X,2), size(X, 1))


function sandwich(x::VceModel, S::Matrix{Float64})
	temp = At_mul_B(x.X, x.X)
	H = inv(cholfact!(temp))
	H * S * H
end

function sandwich(x::VceModelH, S::Matrix{Float64})
	 x.H * S * x.H
end


#
# User can write new vcov in term of first variable
#

StatsBase.vcov(x::LinearModel, v::AbstractVce) = StatsBase.vcov(VceModelH(x.pp.X, residuals(x), inv(cholfact(x))), v)


#
# User can write new vcov in term of second variable
#


immutable type VceSimple <: AbstractVce 
end

StatsBase.vcov(x::VceModel, t::VceSimple) = StatsBase.vcov(x)

function StatsBase.vcov(x::VceModel, t::VceSimple)
	H = At_mul_B(x.X, x.X)
	H = inv(cholfact!(H))
 	scale!(H, sum(x.residuals.^2)/  x.df_residual)
end

function StatsBase.vcov(x::VceModelH, t::VceSimple)
 	x.H * (sum(x.residuals.^2)/  x.df_residual)
end

StatsBase.vcov(x::AbstractVceModel, t::VceSimple, df) = StatsBase.vcov(x, t)


#
# White
#

immutable type VceWhite <: AbstractVce 
end

function StatsBase.vcov(x::AbstractVceModel, t::VceWhite) 
	Xu = broadcast(*,  x.X, x.residuals)
	S = At_mul_B(Xu, Xu)
	scale!(S, x.nobs/x.df_residual)
	sandwich(x, S) 
end

StatsBase.vcov(x::AbstractVceModel, t::VceWhite, df) = StatsBase.vcov(x, t)

#
# HAC
#

immutable type VceHac <: AbstractVce
	time::Symbol
	nlag::Int
	weightfunction::Function
end
VceHac(time, nlag) = VceHac(time, nlag, (i, n) -> 1 - i/(n+1))
allvars2(x::VceHac) = x.time

function StatsBase.vcov(x::AbstractVceModel, v::VceHac, df)
	time = df[v.time]
	nlag = v.nlag
	weights = map(i -> v.weightfunction(i, nlag), [1:nlag])

	Xu = broadcast(*,  x.X, x.residuals)
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
	scale!(S, x.nobs/x.df_residual)
	sandwich(x, S) 
end

function lag(x::Array, n::Int, time::Vector) 
	index = findin(time - n, time)
	lagx = DataArray(eltype(x), dim(x))
	lagx[index] = x[index]
	return(lagx)
end

#
# Cluster
#

immutable type VceCluster  <: AbstractVce
	clusters::Vector{Symbol}
end
VceCluster(x::Symbol) = VceCluster([x])

allvars2(x::VceCluster) = x.clusters
# Cameron, Gelbach, & Miller (2011).
function StatsBase.vcov(x::AbstractVceModel, v::VceCluster, df::DataFrame) 
	df = df[v.clusters]
	S = fill(zero(Float64), (size(x.X, 2), size(x.X, 2)))
	for i in 1:length(v.clusters)
		for c in combinations(v.clusters, i)
			if rem(length(c), 2) == 1
				S += helper_cluster(x, group(df[c]))
			else
				S -= helper_cluster(x, group(df[c]))
			end
		end
	end
	scale!(S, (x.nobs - 1) / x.df_residual)
	sandwich(x, S)
end

function helper_cluster(x::AbstractVceModel, f::PooledDataArray)
	X = x.X
	residuals = x.residuals
	pool = f.pool
	refs = f.refs

	# if only one obs by pool, use White, as in Petersen (2009) & Thomson (2011)
	if length(pool) == size(X, 1)
		Xu = broadcast(*,  x.X, x.residuals)
		At_mul_B(Xu, Xu)
		return(At_mul_B(Xu, Xu))
	else
		# otherwise
		X2 = fill(zero(Float64), (size(X, 2), length(f.pool)))
		for j in 1:size(X, 2)
			for i in 1:size(X, 1)
				X2[j, refs[i]] += X[i, j] * residuals[i]
			end
		end
		out = A_mul_Bt(X2, X2)
		scale!(out, length(pool) / length(pool - 1))
		return(out)
	end
end








