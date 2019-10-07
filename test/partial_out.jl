using FixedEffectModels, DataFrames, Statistics, CSV, Test

df = CSV.read(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))
df.pState = categorical(df.State)
df.pYear = categorical(df.Year)


@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ NDI))[1])[1:5, :] ≈ [ -37.2108  9.72654; -35.5599  9.87628; -32.309   8.82602; -34.2826  9.64653; -35.0526  8.84143] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ NDI + fe(pState)))[1])[1:5, :] ≈ [ -21.3715  -0.642783; -19.6571  -0.540335; -16.3427  -1.63789; -18.2631  -0.856975; -18.9784  -1.70283] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ 1 + fe(pState)))[1])[1:5, :] ≈  [-13.5767   -40.5467; -12.0767   -39.3467; -8.97667  -39.3467; -11.0767   -37.6467; -11.9767   -37.5467] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ 1))[1])[1:5, :] ≈ [ -30.0509  -40.0999; -28.5509  -38.8999; -25.4509  -38.8999; -27.5509  -37.1999; -28.4509  -37.0999] atol = 1e-3
@test mean(convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ NDI), add_mean = true)[1]), dims = 1[1]) ≈ [123.951  68.6999] atol = 1e-3
@test mean(convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ NDI + fe(pState)), add_mean = true)[1]), dims = 1[1]) ≈ [123.951  68.6999] atol = 1e-3
@test mean(convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ 1 + fe(pState)), add_mean = true)[1]), dims = 1[1]) ≈ [123.951  68.6999] atol = 1e-3
@test mean(convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ 1), add_mean = true)[1]), dims = 1[1]) ≈ [123.951  68.6999] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ NDI), weights = :Pop)[1])[1:5, :] ≈[ -37.5296  11.8467; -35.8224  11.9922; -32.5151  10.9377; -34.4416  11.7546; -35.163   10.9459] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ NDI + fe(pState)), weights = :Pop)[1])[1:5, :] ≈ [ -22.2429  -1.2635 ; -20.5296  -1.1515 ; -17.2164  -2.23949; -19.1378  -1.45057; -19.854   -2.28819] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ 1 + fe(pState)), weights = :Pop)[1])[1:5, :] ≈ [ -14.0383   -43.1224; -12.5383   -41.9224; -9.43825  -41.9224; -11.5383   -40.2224; -12.4383   -40.1224] atol = 1e-3
@test convert(Matrix{Float64}, partial_out(df, @formula(Sales + Price ~ 1), weights = :Pop)[1])[1:5, :] ≈ [ -26.3745  -44.9103; -24.8745  -43.7103; -21.7745  -43.7103; -23.8745  -42.0103; -24.7745  -41.9103] atol = 1e-3
    
