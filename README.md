The function `demean` accepts a dataframe, columns (an array of synbols), and a set of grouping variables (an array of an array of columns), and returns a dataframe where vectors in second arguments are replaced by their demeaned version

```
using DataFrames
using RDatasets
using FixedEffects

df = dataset("plm", "Cigar")
result = demean(df, [:Sales], Vector{Symbol}[[:State],[:Year]])
```

Check that  Sales average to zero with respect to State and year

```R
by(result, :State, df -> mean(df[:Sales]))
by(result, :Year, df -> mean(df[:Sales]))
```

If the dataframe contains missing values, new rows are set to missing

```
df[1:5, :Sales] = NA
result = demean(df, [:Sales], Vector{Symbol}[[:State],[:Year]])
```
