using DataFrames, Test, FixedEffectModels


df = DataFrame(v1 = @data(["ok", missing, "ok", "first"]), v2 = @data([1, 2, 3, 1]), v3 = @data([missing, 1, 1, missing]))

@test levels(group(df[:v1])) == [1, 2]
@test group(df[:v1]).refs == [1, 0, 1, 2]

@test group(df[:v2]) == categorical(df[:v2])

@test levels(group(df[:v3])) ==  [1]
@test group(df[:v3]).refs ==  [0, 1, 1, 0]

@test levels(group(df, [:v1, :v2, :v3])) == [1]
@test group(group(df, [:v1, :v2, :v3])).refs == [0, 0, 1, 0]


# test different syntaxes
df = DataFrame(v1 = categorical(collect(1:1000)), v2 = categorical(fill(1, 1000)))
@test group(df, [:v1, :v2]) == collect(1:1000)
@test group(df, :v1, :v2) == collect(1:1000)
@test group(df[:v1], df[:v2]) == collect(1:1000)
