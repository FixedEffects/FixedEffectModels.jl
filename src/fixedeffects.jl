module FixedEffects
using DataFrames, DataArrays

export group, demean!, demean, areg, regife
include("demean.jl")
include("areg.jl")
include("regife.jl")
end