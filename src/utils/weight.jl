
##############################################################################
##
## Create light weight type
## 
##############################################################################

type Ones <: AbstractVector{Float64}
    length::Int
end
Base.size(O::Ones) = O.length
Base.getindex(::Ones, ::Int...) = one(Float64)
# Add in version 0.4 Base.unsafe_getindex(::Ones, ::Int...) = 1
Base.broadcast!{T}(o::Function, ::Array{Float64, T}, ::Array{Float64, T}, ::Ones) = nothing
Base.scale!(::Vector{Float64}, ::Ones) = nothing


get_weight(df::AbstractDataFrame, weight::Symbol) = sqrt(df[weight])
get_weight(df::AbstractDataFrame, ::Nothing) = Ones(size(df, 1))





