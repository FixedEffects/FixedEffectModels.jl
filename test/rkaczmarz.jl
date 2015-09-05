using FixedEffectModels, Base.Test

tol = 1e-10
N=10
M = 100
A = randn(M,N)
x = randn(N)
rhs = A*x

# sparse matrix
(x, converged) = FixedEffectModels.rkaczmarz(A',rhs, tol = tol)
@test norm(A*x - rhs, Inf) <= tol

# full matrix
x,ch = FixedEffectModels.rkaczmarz(full(A'),rhs, tol = tol)
@test norm(A*x - rhs, Inf) <= tol