@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr))
