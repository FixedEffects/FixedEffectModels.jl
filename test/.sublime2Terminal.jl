df[:n] = max.(1:size(df, 1), 60)
df[:pn] = categorical(df[:n])
m = @model y ~ x1 fe = pn  vcov = cluster(pid1)
x = reg(df, m)
@test x.nobs == 60