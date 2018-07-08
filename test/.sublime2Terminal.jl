
using FixedEffectModels, CSV, DataFrames, LinearAlgebra, Test
df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv"))