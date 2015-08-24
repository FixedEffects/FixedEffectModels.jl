
##############################################################################
##
## Create light weight type
## 
##############################################################################

# http://stackoverflow.com/a/30968709/3662288
immutable Ones <: AbstractVector{Float64}
	length::Int
end
Base.size(O::Ones) = O.length
Base.getindex(O::Ones, ::Int...) = one(Float64)
# Add in version 0.4 Base.unsafe_getindex(::Ones, ::Int...) = 1
Base.broadcast!{T}(a::Function, b::Array{Float64, T}, c::Array{Float64, T}, d::Ones) = nothing
Base.scale!(a::Vector{Float64}, b::Ones) = nothing


function get_weight(df::AbstractDataFrame, weight::Symbol)
	w = convert(Vector{Float64}, df[weight])
	sqrtw = sqrt(w)
end
get_weight(df::AbstractDataFrame, ::Nothing) = Ones(size(df, 1))





