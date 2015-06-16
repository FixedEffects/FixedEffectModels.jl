module FixedEffects

export group, demean!, demean, areg, regife

include("demean.jl")
include("areg.jl")
include("group.jl")
include("regife.jl")

end