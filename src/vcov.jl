using GLM, DataFrames

abstract AbstractVcovModel


immutable type VcovModel{T} <: AbstractVcovModel
	X::Matrix{T} # If weight, matrix should be X\sqrt{W}
	residuals::Vector{T}
	df_residual::Integer
	nobs::Integer
end
VcovModel{T}(X::Matrix{T}, residual::Vector{T}) = VcovModel(X, residual, size(X, 1) - size(X,2), size(X, 1))

immutable type VcovModelH{T} <: AbstractVcovModel
	X::Matrix{T} # If weight, matrix should be X\sqrt{W}
	H::Matrix{T} # If weight, matrix should be X\sqrt{W}
	residuals::Vector{T}
	df_residual::Integer
	nobs::Integer
end
VcovModelH{T}(X::Matrix{T}, H::Matrix{T}, residual::Vector{T}) = VcovModelH(X, H, residual, size(X, 1) - size(X,2), size(X, 1))

#
#
# Helper
#
function helper(X::Matrix, H::Matrix, S::Matrix)
	temp = H * S
	if size(temp, 1) > 1
		A_mul_B!(temp, temp, H)
	else
		temp * H
	end
end

function sandwich(x::VcovModel, S::Matrix{Float64})
	H = At_mul_B(x.X, x.X)
	helper(x.X, inv(cholfact!(H)))
end

function sandwich(x::VcovModelH, S::Matrix{Float64})
	helper(x.X, x.H, S)
end

# Simple

function vcov(x::VcovModel)
	H = At_mul_B(x.X, x.X)
	H = inv(cholfact!(H))
 	scale!(H, sum(x.residuals.^2)/  x.df_residual)
end

function vcov(x::VcovModelH)
 	scale!(H, sum(x.residuals.^2)/  x.df_residual)
end

#
# Robust
#

function vcov_robust(x::AbstractVcovModel)
	X2 = broadcast(*,  x.X, x.residuals)
	scale = At_mul_B(X2, X2)
	sandwich(x, scale) * (x.nobs/x.df_residual)
end


#
# clustered
#

function vcov_within(x::AbstractVcovModel, f::PooledDataArray)
	X = x.X
	residuals = x.residuals
	pool = f.pool
	refs = f.refs
	X2 = fill(zero(Float64), (size(X, 2), length(f.pool)))
	for j in 1:size(X, 2)
		for i in 1:size(X, 1)
			X2[j, refs[i]] += X[i, j] * residuals[i]
		end
	end
	A_mul_Bt(X2, X2)
end


function vcov_cluster(x::AbstractVcovModel, f::PooledDataArray)
	scale = vcov_within(x, f)
	out = sandwich(x, scale)
	scale!(out, length(f.pool) / (length(f.pool) - 1) * (x.nobs - 1) / x.df_residual)
end
function vcov_cluster(x::AbstractVcovModel, f1::PooledDataArray, f2::PooledDataArray)
	Xf1 = vcov_within(x, f1)
	Xf2 = vcov_within(x, f2)
	Xf3 = vcov_within(x, group(DataFrame(f1 = f1, f2 = f2)))
	scale = Xf1 + Xf2 - Xf3
	# not sure what df is
	sandwich(x, scale)
end
function vcov_cluster(x::AbstractVcovModel, df::AbstractDataFrame) 
	if size(df, 2) == 1
		vcov_cluster(x, df[1])
	elseif size(df, 2) == 2
		vcov_cluster(x, df[1], df[2])
	else
		error("Can't compute > 2 clusters")
	end
end


# Support for LinearModel
VcovModel(x::LinearModel) = VcovModelH(x.pp.X, residuals(x), inv(cholfact(x)))
vcov_robust(x::LinearModel) = vcov_robust(VcovModelH(x))
vcov_cluster(x::LinearModel, f::PooledDataArray) = vcov_cluster(VcovModel(x), f)
vcov_cluster(x::LinearModel, df::AbstractDataFrame) = vcov_cluster(VcovModel(x), df)
vcov_cluster(x::LinearModel, f1::PooledDataArray, f2::PooledDataArray) = vcov_cluster(VcovModel(x), f1, f2)








