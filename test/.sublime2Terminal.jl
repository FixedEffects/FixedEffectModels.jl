m = @model y ~ x1 fe = pn  vcov = cluster(pid1)
x = reg(df, m, drop_singletons = false)