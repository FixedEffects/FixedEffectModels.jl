df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset", "EmplUK.csv"))
df[:id1] = df[:Firm]
df[:id2] = df[:Year]
df[:pid1] = categorical(df[:id1])
df[:pid2] = categorical(df[:id2])
df[:y] = df[:Wage]
df[:x1] = df[:Emp]
df[:w] = df[:Output]
