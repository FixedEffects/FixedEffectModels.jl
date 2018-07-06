using FixedEffectModels, CSV, DataFrames, Test
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

##############################################################################
##
## coefficients 
##
##############################################################################

# simple
m = @model y ~ x1
x = reg(df, m)
@test coef(x) ≈ [139.73446,-0.22974] atol = 1e-4
m = @model y ~ x1 weights = w
x = reg(df, m)
@test coef(x) ≈ [137.72495428982756,-0.23738] atol = 1e-4

df[:SalesInt] = round.(Int64, df[:Sales])
m = @model SalesInt ~ Price
x = reg(df, m)
@test coef(x) ≈ [139.72674,-0.2296683205] atol = 1e-4



# absorb
m = @model y ~ x1 fe = pid1
x = reg(df, m)
@test coef(x) ≈ [-0.20984] atol = 1e-4
@test x.iterations == 1

m = @model y ~ x1 fe = pid1 + pid2
x = reg(df, m)
@test coef(x)  ≈ [-1.08471] atol = 1e-4
m = @model y ~ x1 fe = pid1 + pid1&id2
x = reg(df, m)
@test coef(x) ≈   [-0.53470] atol = 1e-4
m = @model y ~ x1 fe = pid1*id2
x = reg(df, m)
@test coef(x) ≈   [-0.53470] atol = 1e-4
#@test isempty(coef(reg(df, @formula(y ~ 0), @fe(pid1*x1))))
m = @model y ~ x1 fe = ppid1&pid2
x = reg(df, m)
@test coef(x)  ≈  [-1.44255] atol = 1e-4

m = @model y ~ x1 fe = pid1&id2
x = reg(df, m)
@test coef(x) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
m = @model y ~ x1 fe = id2&pid1
x = reg(df, m)
@test coef(x) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
m = @model y ~ 1 fe = id2&pid1
x = reg(df, m)
@test coef(x) ≈ [174.4084407796102] atol = 1e-4

m = @model y ~ x1 fe = pid1&id2 + pid2&id1
x = reg(df, m)
@test coef(x) ≈ [51.2359,- 0.5797] atol = 1e-4
m = @model y ~ x1 + x2 fe = pid1&id2 + pid2&id1
x = reg(df, m)
@test coef(x) ≈ [-46.4464,-0.2546, -0.005563] atol = 1e-4
m = @model y ~ 0 + x1 + x2 fe = pid1&id2 + pid2&id1
x = reg(df, m)
@test coef(x) ≈  [-0.21226562244177932,-0.004775616634862829] atol = 1e-4


# recheck these two below
m = @model y ~ z1 fe = x1&x2&pid1
x = reg(df, m)
@test coef(x)  ≈  [122.98713, 0.30933] atol = 1e-4
m = @model y ~ z1 fe = (x1&x2)*pid1
x = reg(df, m)
@test coef(x) ≈   [ 0.421406] atol = 1e-4

# only one intercept
m = @model y ~ 1 fe = pid1 + pid2
x = reg(df, m)

# absorb + weights
m = @model y ~ x1 fe = pid1 weights = w
x = reg(df, m)
@test coef(x) ≈ [- 0.21741] atol = 1e-4
m = @model y ~ x1 fe = pid1 + pid2 weights = w
x = reg(df, m)
@test coef(x) ≈ [- 0.88794] atol = 1e-3
m = @model y ~ x1 fe = pid1 + pid1&id2 weights = w
x = reg(df, m)
@test coef(x) ≈ [- 0.461085492] atol = 1e-4

# iv
m = @model y ~ (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈ [138.19479,- 0.20733] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈  [137.45096,0.00516,- 0.76276] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [137.57335,0.00534,- 0.78365] atol = 1e-4
## multiple endogeneous variables
m = @model y ~ (x1 + x2 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [139.544, .8001, -.00937] atol = 1e-4
m = @model y ~ 1 + (x1 + x2 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [139.544, .8001, -.00937] atol = 1e-4
result = [196.576, 0.00490989, -2.94019, -3.00686, -2.94903, -2.80183, -2.74789, -2.66682, -2.63855, -2.52394, -2.34751, -2.19241, -2.18707, -2.09244, -1.9691, -1.80463, -1.81865, -1.70428, -1.72925, -1.68501, -1.66007, -1.56102, -1.43582, -1.36812, -1.33677, -1.30426, -1.28094, -1.25175, -1.21438, -1.16668, -1.13033, -1.03782]
m = @model y ~ x2 + (x1&pid2 ~ z1&pid2)
x = reg(df, m)
@test coef(x) ≈ result atol = 1e-4


# iv + weight
m = @model y ~ (x1 ~ z1) weights = w
x = reg(df, m)
@test coef(x) ≈ [137.03637,- 0.22802] atol = 1e-4


# iv + weight + absorb
m = @model y ~ (x1 ~ z1) fe = pid1
x = reg(df, m)
@test coef(x) ≈ [-0.20284] atol = 1e-4
m = @model y ~ (x1 ~ z1) fe = pid1 weights = w
x = reg(df, m)
@test coef(x) ≈ [-0.20995] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈  [137.45096580480387,0.005169677634275297,-0.7627670265757879] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1) fe = pid1
x = reg(df, m)
@test coef(x)  ≈  [0.0011021722526916768,-0.3216374943695231] atol = 1e-4



# non high dimensional factors
m = @model y ~ x1 + pid2
x = reg(df, m)
m = @model y ~  pid2 fe = pid1