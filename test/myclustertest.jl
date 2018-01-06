# mytest.jl
using DataFrames, FixedEffectModels, Base.Test
#df = readtable("/Users/Matthieu/Dropbox/Github/FixedEffectModels.jl/dataset/Cigar.csv.gz")
#df = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv.gz"))
df = readtable("/Users/jboehm/.julia/FixedEffectModels/dataset/asi.csv")

df[:state] = pool(df[:state])
df[:numid] = pool(df[:numid])
df[:year] = pool(df[:year])

m = @model gsv ~ qmanufactured fe = numid + year vcov = cluster(state)
x = reg(df, m)

#
# m = @model y ~ x1 vcov = cluster(pid1)
# x = reg(df, m)
# @test stderr(x)[2] ≈ 0.03792 atol = 1e-4
# m = @model y ~ x1 fe = pid1 vcov = cluster(pid2)
# x = reg(df, m)
# @test stderr(x) ≈ [0.02205] atol = 1e-4
# # stata reghxe
# m = @model y ~ x1 fe = pid1 vcov = cluster(pid1)
# x = reg(df, m)
# @test stderr(x) ≈ [0.03573] atol = 1e-4
#
# # no reference
# m = @model y ~ x1 vcov = cluster(pid1 + pid2)
# x = reg(df, m)
# @test stderr(x)[1] ≈ 6.17025 atol = 1e-4
# # no reference
# m = @model y ~ x1 fe = pid1 vcov = cluster(pid1 + pid2)
# x = reg(df, m)
# @test stderr(x)[1] ≈ 0.04037 atol = 1e-4
