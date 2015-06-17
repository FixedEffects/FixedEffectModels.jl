module FixedEffects

export group, demean!, demean, areg, regife, ErrorModel, vcov, vcov_robust, vcov_cluster, vcov_cluster2

include("demean.jl")
include("areg.jl")
include("group.jl")
include("regife.jl")
include("error.jl")

end