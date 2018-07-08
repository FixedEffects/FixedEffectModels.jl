@time reg(df, @model(y ~ x1 + x2, fe = id1, weights = w))
