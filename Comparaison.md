
Julia

```julia
N = 1000000
K = 10000
df = DataFrame(
  v1 =  rand(1:N, N),
  v2 =  rand(1:K, N),
  v3 =  randn(N), # numeric e.g. 23.5749
  v4 =  randn(N) # numeric e.g. 23.5749
)
@time FixedEffects.demean(df, [:v3,:v4], Vector{Symbol}[[:v1]])
#elapsed time: 2.783390545 seconds (1131099264 bytes allocated, 24.41% gc time)
1000000x6 DataFrame
@time FixedEffects.demean(df, [:v3,:v4], Vector{Symbol}[[:v1],[:v2]])
# elapsed time: 17.700998963 seconds (7270095584 bytes allocated, 29.14% gc time)
1000000x6 DataFrame
```

R

```R
library(lfe)
N = 1000000
K = N/100
df = data_frame(
  v1 =  sample(N, N, replace = TRUE),
  v2 =  sample(K, N, replace = TRUE),
  v3 =  runif(N), # numeric e.g. 23.5749
  v4 =  runif(N) # numeric e.g. 23.5749
)
system.time(felm(v3+v4~1|v1, df))
 user  system elapsed 
3.909   0.117   4.009 
system.time(felm(v3+v4~1|v1+v2, df))
 user  system elapsed 
5.009   0.147   4.583 
```