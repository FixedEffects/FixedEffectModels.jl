
using FixedEffectModels, CSV, DataFrames, LinearAlgebra, Test
df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv"))
df[:id1] = df[:State]
df[:id2] = df[:Year]
df[:pid1] = categorical(df[:id1])
df[:ppid1] = categorical(div.(df[:id1], 10))
df[:pid2] = categorical(df[:id2])
df[:y] = df[:Sales]
df[:x1] = df[:Price]
df[:z1] = df[:Pimin]
df[:x2] = df[:NDI]
df[:w] = df[:Pop]