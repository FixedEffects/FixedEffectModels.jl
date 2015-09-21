using DataArrays, DataFrames, FixedEffectModels, Base.Test


df = DataFrame(v1 = @data(["ok", NA, "ok", "first"]), v2 = @data([1, 3, 2, 1]), v3 = @data([NA, 1, 1, NA]))

@test group(df[[:v1]]).pool == [1, 2]
@test group(df[[:v1]]).refs == [2, 0, 2, 1]

@test group(df[[:v2]]) == pool(df[:v2])
@test group(df[[:v2]]) == pool(df[:v2])

@test group(df[[:v3]]).pool ==  [1]
@test group(df[[:v3]]).refs ==  [0, 1, 1, 0]

@test group(df).pool == [1]
@test group(df).refs == [0, 0, 1, 0]


# test different syntaxes
df = DataFrame(v1 = pool(collect(1:1000)), v2 = pool(fill(1, 1000)))
@test group(df, [:v1, :v2]) == collect(1:1000)
@test group(df, :v1, :v2) == collect(1:1000)
@test group(df[:v1], df[:v2]) == collect(1:1000)
