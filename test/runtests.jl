using FixedEffectModels, Test

@testset "formula" begin include("formula.jl") end
@testset "fit" begin include("fit.jl") end
@testset "predict" begin include("predict.jl") end
@testset "partial out" begin include("partial_out.jl") end
@testset "collinearity" begin include("collinearity.jl") end
