using DataFrames, CSV, FixedEffectModels, Random, Statistics, Test, LinearAlgebra
using FixedEffectModels: reinsert_omitted!

@testset "collinearity_with_fixedeffects" begin

  # read the data
  csvfile = CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))
  df = DataFrame(csvfile)

  # create a bigger dataset
  sort!(df, [:State, :Year])
  nstates = maximum(df.State)
  dflarge = copy(df)
  for i in 1:100
      dfnew = copy(df)
      dfnew.State .+= i .* nstates
      append!(dflarge, dfnew)
  end

  # create a dummy variable that is collinear with the State-FE
  dflarge.highstate = dflarge.State .< median(dflarge.State)

  # create a second 'high-dimensional' categorical variable
  Random.seed!(1234)
  dflarge.catvar = rand(1:200, nrow(dflarge))

  # run the regression with the default setting (tol = 1e-6)
  rr = reg(dflarge, @formula(Price ~ highstate + Pop + fe(Year) + fe(catvar) + fe(State)), Vcov.cluster(:State); tol=1e-6)

  # test that the collinear coefficient is zero and the standard error is NaN
  @test rr.coef[1] ≈ 0.0
  @test isnan(stderror(rr)[1])



  df = DataFrame(
      [36.9302  44.5105;
       39.4935  44.5044;
       38.946   44.5072;
       37.8005  44.5098;
       37.2613  44.5103;
       35.3885  44.5109;], 
       :auto
  )
  rr = reg(df, @formula(x1 ~ x2))
  @test all(!isnan, stderror(rr))
end

@testset "reinsert omitted" begin
  coef = [10.0, 20.0]
  vcov = Symmetric([1.0 0.2; 0.2 4.0])
  basis_coef = BitVector([true, false, true])
  perm = [2, 1, 3]

  newcoef, newvcov = reinsert_omitted!(coef, vcov, basis_coef, perm)
  newvcov_matrix = Matrix(newvcov)

  @test newcoef == [0.0, 10.0, 20.0]
  @test isnan(newvcov_matrix[1, 1])
  @test isnan(newvcov_matrix[1, 2])
  @test isnan(newvcov_matrix[1, 3])
  @test isnan(newvcov_matrix[2, 1])
  @test newvcov_matrix[2, 2] == 1.0
  @test newvcov_matrix[2, 3] == 0.2
  @test isnan(newvcov_matrix[3, 1])
  @test newvcov_matrix[3, 2] == 0.2
  @test newvcov_matrix[3, 3] == 4.0
end

@testset "small-scale independent regressor is kept" begin
  # The rank test is scale-invariant: a genuinely independent regressor whose total
  # sum of squares is below sqrt(eps) used to be dropped as collinear (NaN stderror).
  df = DataFrame(CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv")))
  df.tiny = df.Price .* 3e-8
  m = reg(df, @formula(Sales ~ tiny))
  mP = reg(df, @formula(Sales ~ Price))
  @test !isnan(stderror(m)[2])
  # scaling a regressor by c scales its coefficient by 1/c and leaves the t-stat invariant
  @test coef(m)[2] * 3e-8 ≈ coef(mP)[2] rtol = 1e-4
  @test coef(m)[2] / stderror(m)[2] ≈ coef(mP)[2] / stderror(mP)[2] rtol = 1e-4
end

@testset "collinear regressor reported as zero coef and NaN stderror" begin
  df = DataFrame(y = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0], x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  df.x2 = 2 .* df.x        # perfectly collinear with x
  m = reg(df, @formula(y ~ x + x2))
  # coefnames are [(Intercept), x, x2]; the later collinear column x2 is dropped
  @test coef(m)[3] == 0
  @test isnan(stderror(m)[3])
  @test !isnan(stderror(m)[2])
end

@testset "fixed-effect slopes omit matching RHS terms" begin
  df = DataFrame(
    id1 = repeat(1:3, inner = 4),
    id2 = repeat(1:2, inner = 6),
    x1 = collect(1.0:12.0),
    x2 = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
  )
  df.y = df.x1 .* repeat([1.0, 2.0, 3.0], inner = 4) .+
         df.x2 .* repeat([0.5, -0.5], inner = 6) .+ sin.(df.x1)

  m = @test_logs min_level=Base.CoreLogging.Info reg(
    df, @formula(y ~ fe(id1)*x1 + fe(id2)*x2); progress_bar = false)
  @test isempty(coefnames(m))
  @test isempty(coef(m))

  # Unrelated, data-driven collinearity with an intercept FE is still reported.
  df.group_constant = Float64.(df.id1)
  @test_logs (:info, r"RHS-variable group_constant is collinear with the fixed effects") reg(
    df, @formula(y ~ group_constant + fe(id1)); progress_bar = false)
end
