

The function `demean` is a basic implementation of the function `demean` in the [lfe R package](http://journal.r-project.org/archive/2013-2/gaure.pdf). It allows to estimate models with multiple high dimentional fixed effects.

The function `demean` accepts a dataframe, a set of columns to demean (an array of symbols), and a set of grouping variables (an array of an array of symbols). It returns a new data.frame with the demeaned version of columns.

For instance, the following command returns the residuals of the regression of Sales on State dummies and Year dummies.

```julia
using DataFrames
using RDatasets

df = dataset("plm", "Cigar")
result = FixedEffects.demean(df, :Sales, Vector{Symbol}[[:State],[:Year]])
```


If the dataframe contains missing values, new rows are set to missing

```julia
df[1:5, :Sales] = NA
result = FixedEffects.demean(df, [:Sales], Vector{Symbol}[[:State],[:Year]])
```



