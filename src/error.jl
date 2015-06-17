# model.frame is a deep copy due to na_omit
# model matrix is another deep copy
# e(sample) is result.mf.msng

# why is model frame stocked? one could (i) store sub instead of na_omit (ii) just do df[e(sample)]


# result has model, mm, mf

# result.model  is a LinPredModel

# result.model.pp  is a LinPred
## X, beta0., qr (which has factors, T)
# A LinPred type must incorporate some form of a decomposition of the weighted model matrix that allows for the solution of a system X'W * X * delta=X'wres where W is a diagonal matrix of "X weights", provided as a vector of the square roots of the diagonal elements, and wres is a weighted residual vector.

# result.model.rr is a LMresp
# field mu (mean response), offset, wts (weugt), y (response)


# start with X, res

using GLM, DataFrames

type ErrorModel{T}
	X::Matrix{T}
	residual::Vector{T}
end

ErrorModel(x::LinPredModel) = ErrorModel(x.pp.X, residuals(x))

function helper(X, S)
	H = At_mul_B(X, X)
	H = inv(cholfact!(H))
    H * S * H
end

function vcov_within(X, residual, f)
	pool = f.pool
	refs = f.refs
	X2 = fill(zero(Float64), (size(X, 2), length(f.pool)))
	for j in 1:size(X, 2)
		for i in 1:size(X, 1)
			X2[j, refs[i]] += X[i, j] * residual[i]
		end
	end
	A_mul_Bt(X2, X2)
end


function vcov(x::ErrorModel)
	H = At_mul_B(x.X, x.X)
	H = inv(cholfact!(H))
 	scale!(H, sum(x.residual.^2)/(length(x.residual)))
 	H
 end

function vcov_robust(x::ErrorModel)
	X2 = broadcast(*,  x.X, x.residual)
	scale = At_mul_B(X2, X2)
	helper(x.X, scale)
end

function vcov_cluster(x::ErrorModel, f::PooledDataArray)
	scale = vcov_within(x.X, x.residual, f)
	helper(x.X, scale)
end

function vcov_cluster2(x::ErrorModel, f1::PooledDataArray, f2::PooledDataArray)
	Xf1 = vcov_within(x.X, x.residual, f1)
	Xf2 = vcov_within(x.X, x.residual, f2)
	Xf3 = vcov_within(x.X, x.residual, group(DataFrame(f1 = f1, f2 = f2)))
	scale = Xf1 + Xf2 - Xf3
	helper(x.X, scale)
end




