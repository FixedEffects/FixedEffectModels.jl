This is an experimental function.

The function `demean` accepts a dataframe, columns (an array of synbols), and a set of grouping variables (an array of an array of columns), and returns new demeaned columns with the suffix `_p`.


```julia
using DataFrames
using RDatasets

df = dataset("plm", "Cigar")
result = FixedEffects.demean(df, [:Sales], Vector{Symbol}[[:State],[:Year]])
```

Check that  Sales average to zero with respect to State and year

```julia
by(result, :State, df -> mean(df[:Sales]))
by(result, :Year, df -> mean(df[:Sales]))
```

If the dataframe contains missing values, new rows are set to missing

```julia
df[1:5, :Sales] = NA
result = FixedEffects.demean(df, [:Sales], Vector{Symbol}[[:State],[:Year]])
```
