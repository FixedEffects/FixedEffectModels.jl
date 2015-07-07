using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

result = reg(Sales ~ Price, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9, maxiter = 10000)

# test coef
@test_approx_eq_eps result.coef [163.0135038895678,-0.4061036361588258] 1e-8
# check normalization F'F/T = Id and Lambda' Lambda = diag
@test_approx_eq transpose(result.factors) * result.factors  30 * eye(size(result.factors, 2))
@test_approx_eq_eps (transpose(result.loadings)* result.loadings - diagm(diag(transpose(result.loadings)* result.loadings))) fill(zero(Float64), (size(result.loadings, 2), size(result.loadings, 2))) 1e-8


# test coef with absorb option
@test_approx_eq_eps reg(Sales ~ Price |> pState, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9).coef  [-0.42538935900021146] 1e-8




#beta1 =  1
#beta2 =  3
#mu =  5
#mu1 =  1
#mu2 =  1
#gamma =  2
#delta =  4
#N = 1000
#T = 50
#
#
#
#lambda1 = randn(N)
#lambda2 = randn(N)
#x = lambda1 + lambda2 + rand(N)
#
#F1 = randn(T)
#F2 = randn(T)
#w = F1 + F2 
#
#
#X1 = mu1 + lambda1 * F1' + lambda2 * F2' +  lambda1 * fill(1, T)' + lambda2 * fill(1, T)'+ fill(1, N) * F1' + fill(1,# N) * F2' + rand(N,T)
#X2 = mu2 + lambda1 * F1' + lambda2 * F2' +  lambda1 * fill(1, T)' + lambda2 * fill(1, T)'+ fill(1, N) * F1' + fill(1,# N) * F2' + rand(N,T)
#Y = mu + beta1 * X1 + beta2 * X2  + gamma * x  * fill(1, T)' + delta *fill(1, N) * w' +  lambda1 * F1' + lambda2 * F2#' + rand(N, T)
#
#
#vY = vec(Y)
#vX1 = vec(X1)
#vX2 = vec(X2)
#
#v = [1:N;]
#vN = [v[div(i,T)+1] for i=0:(T*N-1)]
#vx = [x[div(i,T)+1] for i=0:(T*N-1)]
#
#v = [1:T;]
#vT = [v[mod(i, T)+1] for i=0:(T*N-1)]
#vw = [w[mod(i, T)+1] for i=0:(T*N-1)]
#
#
#df = DataFrame(Y = vY, X1 = vX1, X2 = vX2, x = vx, w = vw, N = vN, T = vT)
#df[:N] = pool(df[:N])
#df[:T] = pool(df[:T])
#
#reg(Y~X1 + X2 + w + x, df, InteractiveFixedEffectModel(:N, :T, 2), tol = 1e-11).iterations#