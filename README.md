

The function `demean` accepts a dataframe, columns (an array of synbols), and a set of grouping variables (an array of an array of symbols). It returns new columns, with the suffix `p`, corresponding to demeaned column.

They correspond to residuals in the model

```
x_i = \Sum_k gamma_{g_k(i) + \epsilon_{i}
```



```julia
using DataFrames
using RDatasets

df = dataset("plm", "Cigar")
result = FixedEffects.demean(df, :Sales, Vector{Symbol}[[:State],[:Year]])
```



The new Sales_p column averages to zero with respect to both state and year

```julia
by(result, :State, result -> mean(result[:Sales_p]))
by(result, :Year, result -> mean(result[:Sales_p]))
```

If the dataframe contains missing values, new rows are set to missing

```julia
df[1:5, :Sales] = NA
result = FixedEffects.demean(df, [:Sales], Vector{Symbol}[[:State],[:Year]])
```

