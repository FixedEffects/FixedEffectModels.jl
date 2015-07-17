##############################################################################
##
## VcovData (and its children) has four important methods: 
##
##############################################################################

type VcovData{N} 
	invcrossmatrix::Matrix{Float64}   # (X'X)^{-1} in the simplest case 
	regressors::Matrix{Float64}       # X
	residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
	df_residual::Int64
	function VcovData(invcrossmatrix::Matrix{Float64}, regressors::Matrix{Float64}, residuals::Array{Float64, N}, 	df_residual::Int64)
		size(regressors, 1) == size(residuals, 1) || error("regressors and residuals should have same  number of rows")
		size(invcrossmatrix, 1) == size(invcrossmatrix, 2) || error("invcrossmatrix is a square matrix")
		size(invcrossmatrix, 1) == (size(regressors, 2) * size(residuals, 2))  || error("invcrossmatrix should be square matrix of dimension size(regressors, 2) x size(residuals, 2)")
		new(invcrossmatrix, regressors, residuals, df_residual)
	end
end
nobs(x::VcovData) = size(x.regressors, 1)


typealias VcovDataVector VcovData{1} 
typealias VcovDataMatrix VcovData{2} 

# convert a linear model into VcovData
function VcovData(x::LinearModel) 
	VcovData(inv(cholfact(x)), x.pp.X, x.residuals, size(x.pp.X, 1))
end


##############################################################################
##
## AbstractVcovMethod (and its children) has two methods: 
## allvars that returns variables needed in the dataframe
## shat! returns a S hat matrix. It may change regressors in place
## vcov! returns a covariance matrix
##
##############################################################################

abstract AbstractVcovMethod
allvars(x::AbstractVcovMethod) = Symbol[]
abstract AbstractVcovMethodData


#
# simple standard errors
#

type VcovSimple <: AbstractVcovMethod end
type VcovSimpleData <: AbstractVcovMethodData end
VcovMethodData(v::VcovSimple, df::AbstractDataFrame) = VcovSimpleData()
function vcov!(v::VcovSimpleData, x::VcovData)
 	scale!(x.invcrossmatrix, abs2(norm(x.residuals, 2)) /  x.df_residual)
end
function shat!(v::VcovSimpleData, x::VcovData)
 	scale(inv(x.invcrossmatrix), abs2(norm(x.residuals, 2)))
end


#
# White standard errors
#

type VcovWhite <: AbstractVcovMethod end
type VcovWhiteData <: AbstractVcovMethodData end
VcovMethodData(v::VcovWhite, df::AbstractDataFrame) = VcovWhiteData()
function vcov!(v::VcovWhiteData, x::VcovData) 
	S = shat!(v, x)
	sandwich(x.invcrossmatrix, S) 
end

function shat!(v::VcovWhiteData, x::VcovData{1}) 
	X = x.regressors
	res = x.residuals
	Xu = scale!(res, X)
	S = At_mul_B(Xu, Xu)
	scale!(S, nobs(x)/x.df_residual)
end

function shat!(t::VcovWhiteData, x::VcovData{2}) 
	X = x.regressors
	res = x.residuals
	dim = size(X, 2) * size(res, 2)
	S = fill(zero(Float64), (dim, dim))
	temp = similar(S)
	kronv = fill(zero(Float64), dim)
	@inbounds for i in 1:nobs(x)
		j = 0
		for l in 1:size(res, 2)
			for k in 1:size(X, 2)
				j += 1
				kronv[j] = X[i, k] * res[i, l]
			end
		end
		temp = A_mul_Bt!(temp, kronv, kronv)
		S += temp
	end
	scale!(S, nobs(x)/x.df_residual)
	return(S)
end


function sandwich(H::Matrix{Float64}, S::Matrix{Float64})
	H * S * H
end



#
# Clustered standard errors
#

type VcovCluster  <: AbstractVcovMethod
	clusters::Vector{Symbol}
end
VcovCluster(x::Symbol) = VcovCluster([x])
allvars(x::VcovCluster) = x.clusters

type VcovClusterData <: AbstractVcovMethodData
	clusters::DataFrame
	size::Dict{Symbol, Int64}
end

function VcovMethodData(v::VcovCluster, df::AbstractDataFrame) 
	vclusters = DataFrame(Vector, size(df, 1), length(v.clusters))
	names!(vclusters, v.clusters)
	vsize = Dict{Symbol, Int64}()
	for c in v.clusters
		p = df[c]
		typeof(p) <: PooledDataArray || error("Cluster variable $(c) is of type $(typeof(p)), but should be a PooledDataArray.")
		vclusters[c] = p
		# may be subset / NA
		vsize[c] = length(unique(p.refs))
	end
	VcovClusterData(vclusters, vsize)
end

function vcov!(v::VcovClusterData, x::VcovData)
	S = shat!(v, x)
	sandwich(x.invcrossmatrix, S)
end
function shat!(v::VcovClusterData, x::VcovData{1}) 
	# Cameron, Gelbach, & Miller (2011).
	clusternames = names(v.clusters)
	X = x.regressors
	Xu = scale!(x.residuals,  X)
	S = fill(zero(Float64), (size(X, 2), size(X, 2)))
	for i in 1:length(clusternames)
		for c in combinations(clusternames, i)
			if length(c) == 1
				f = (v.clusters)[c[1]]
				# no need to group in this case
				fsize = (v.size)[c[1]]
			else
				df = v.clusters[c]
				f = group(df)
				fsize = length(f.pool)
			end
			if rem(length(c), 2) == 1
				S += helper_cluster(Xu, f, fsize)
			else
				S -= helper_cluster(Xu, f, fsize)
			end
		end
	end
	scale!(S, (nobs(x)-1) / x.df_residual)
	return(S)
end



function helper_cluster(Xu::Matrix{Float64}, f::PooledDataArray, fsize::Int64)
	refs = f.refs
	if fsize == size(Xu, 1)
		# if only one obs by pool, use White, as in Petersen (2009) & Thomson (2011)
		return(At_mul_B(Xu, Xu))
	else
		# otherwise
		X2 = fill(zero(Float64), (fsize, size(Xu, 2)))
		for j in 1:size(Xu, 2)
			 @inbounds @simd for i in 1:size(Xu, 1)
				X2[refs[i], j] += Xu[i, j]
			end
		end
		out = At_mul_B(X2, X2)
		scale!(out, fsize / (fsize- 1))
		return(out)
	end
end




function shat!(v::VcovClusterData, x::VcovData{2}) 
	# Cameron, Gelbach, & Miller (2011).
	clusternames = names(v.clusters)
	X = x.regressors
	res = x.residuals
	dim = (size(X, 2) *size(res, 2))
	S = fill(zero(Float64), (dim, dim))
	for i in 1:length(clusternames)
		for c in combinations(clusternames, i)
			if length(c) == 1
				f = (v.clusters)[c[1]]
				# no need to group in this case
				fsize = (v.size)[c[1]]
			else
				df = v.clusters[c]
				f = group(df)
				fsize = length(f.pool)
			end
			if rem(length(c), 2) == 1
				S += helper_cluster(X, res, f, fsize)
			else
				S -= helper_cluster(X, res, f, fsize)
			end
		end
	end
	scale!(S, (nobs(x)-1) / x.df_residual)
	return(S)
end

function helper_cluster(X::Matrix{Float64}, res::Matrix{Float64}, f::PooledDataArray, fsize::Int64)
	refs = f.refs
	dim = size(X, 2) * size(res, 2)
	if fsize == size(X, 1)
		S = fill(zero(Float64), (dim, dim))
		temp = similar(S)
		kronv = fill(zero(Float64), dim)
		@inbounds for i in 1:size(X, 1)
			j = 0
			 for l in 1:size(res, 2)
				for k in 1:size(X, 2)
					j += 1
					kronv[j] = X[i, k] * res[i, l]
				end
			end
			temp = A_mul_Bt!(temp, kronv, kronv)
			S += temp
		end
	else
		# otherwise
		kronv = fill(zero(Float64), fsize, dim)
		@inbounds for i in 1:size(X, 1)
			j = 0
			 for l in 1:size(res, 2)
				for k in 1:size(X, 2)
					j += 1
					kronv[refs[i], j] += X[i, k] * res[i, l]
				end
			end
		end
		S = At_mul_B(kronv, kronv)
		scale!(S, fsize / (fsize- 1))
	end
	return(S)
end









##############################################################################
##
## The following function follows the command ranktest (called by ivreg2)
## RANKTEST: Stata module to test the rank of a matrix using the Kleibergen-Paap rk statistic
## Authors: Frank Kleibergen, Mark E Schaffer
## IVREG2: Stata module for extended instrumental variables/2SLS and GMM estimation
## Authors: Christopher F Baum, Mark E Schaffer, Steven Stillman
## More precisely, it corresponds to the Stata command:  ranktest  (X) (Z), wald full
##############################################################################

function rank_test!(X::Matrix{Float64}, Z::Matrix{Float64}, Pi::Matrix{Float64}, vcov_method_data::AbstractVcovMethodData, df_small::Int, df_absorb::Int)

	K = size(X, 2) 
	L = size(Z, 2) 

	crossz = cholfact!(At_mul_B(Z, Z), :L)
	crossx = cholfact!(At_mul_B(X, X), :L)

	Fmatrix = crossz[:L] 
	Gmatrix = inv(crossx[:L])
	theta = A_mul_Bt(At_mul_B(Fmatrix, Pi),  Gmatrix)

	svd = svdfact(theta, thin = false) 
	u = svd.U
	vt = svd.Vt

	# compute lambda
	if K == 1
		a_qq = sqrtm(A_mul_Bt(u, u))
		b_qq = sqrtm(A_mul_Bt(vt, vt)) 
	else
	    u_12 = u[1:(K-1),(K:L)]
	    v_12 = vt[1:(K-1),K]
	    u_22 = u[(K:L),(K:L)]
	    v_22 = vt[K,K]
	    a_qq = vcat(u_12, u_22) * (u_22 \ sqrtm(A_mul_Bt(u_22, u_22)))
	    b_qq = sqrtm(A_mul_Bt(v_22, v_22)) * (v_22' \ vcat(v_12, v_22)')
	end
	kronv = kron(b_qq, a_qq')
	lambda = kronv * vec(theta)

	# compute vhat
	if typeof(vcov_method_data) == VcovSimpleData
		vhat= eye(L*K) / size(X, 1)
	else
		temp1 = convert(Matrix{eltype(Gmatrix)}, Gmatrix')
		temp2 = inv(crossz[:L])'
		temp2 = convert(Matrix{eltype(temp2)}, temp2)
		k = kron(temp1, temp2)'
		vcovmodel = VcovData{2}(k, Z, X, size(Z, 1) - df_small - df_absorb) 
		matrix_vcov2 = shat!(vcov_method_data, vcovmodel)
		vhat = A_mul_Bt(k * matrix_vcov2, k) 
	end

	# return statistics
	vlab = cholfact!(A_mul_Bt(kronv * vhat, kronv))
	r_kp = lambda' * (vlab \ lambda)
	p_kp = ccdf(Chisq((L-K+1 )), r_kp[1])
	F_kp = r_kp[1] / size(Z, 2)
	return(F_kp, p_kp)
end


