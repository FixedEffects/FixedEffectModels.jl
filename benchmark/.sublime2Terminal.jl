@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7, fe = id1 + id2, subset = x3 .>= 0.5))
