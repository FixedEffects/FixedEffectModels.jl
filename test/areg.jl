# values checked from reghdfe
df = dataset("plm", "Cigar")
df[:State] = PooledDataArray(df[:State])
@test_approx_eq  coef(areg(Sales~NDI, df, nothing ~ State))   -0.0017046786439408937
@test_approx_eq  coef(areg(Sales~NDI, df, nothing ~ State + State&Year))  -0.005686067588968152
@test_approx_eq  coef(areg(Sales~NDI, df, nothing ~  State&Year)) -0.007652680961637854
df[:Year] = PooledDataArray(df[:Year])
@test_approx_eq  coef(areg(Sales~NDI, df, nothing ~ State + Year))   -0.00684384541298097

