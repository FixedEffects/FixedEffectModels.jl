using FixedEffectModels, Test

@testset "formula" include("formula.jl")
@testset "fit" include("fit.jl")
@testset "predict" include("predict.jl")
@testset "partial out" include("partial_out.jl")
