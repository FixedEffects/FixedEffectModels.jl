using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

result = reg(Sales ~ Price, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9, maxiter = 10000)

@test_approx_eq_eps result.coef [163.0135038895678,-0.4061036361588258] 1e-8
# check normalization F'F/T = Id and Lambda' Lambda = diag
@test_approx_eq result.ft * transpose(result.ft)  30*eye(size(result.ft, 1))
@test_approx_eq_eps (transpose(result.lambda)* result.lambda - diagm(diag(transpose(result.lambda)* result.lambda))) fill(zero(Float64), (size(result.lambda, 2), size(result.lambda, 2))) 1e-8


# absorb option
@test_approx_eq_eps reg(Sales ~ Price |> pState, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9).coef  [-0.42538935900021146] 1e-8



