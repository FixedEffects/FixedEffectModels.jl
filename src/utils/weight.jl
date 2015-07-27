
##############################################################################
##
## Create light weight type
## 
##############################################################################

# http://stackoverflow.com/a/30968709/3662288
immutable type Ones <: AbstractVector{Float64}
	length::Int
end
Base.size(O::Ones) = O.length
Base.getindex(O::Ones, I::Int...) = one(Float64)
# Add in version 0.4 Base.unsafe_getindex(O::Ones, I::Int...) = 1
Base.broadcast!(op::Function, X::Matrix{Float64}, Y::Matrix{Float64}, O::Ones) = nothing
Base.broadcast!(op::Function, X::Vector{Float64}, Y::Vector{Float64}, O::Ones) = nothing
Base.scale!(X::Vector{Float64}, O::Ones) = nothing


function get_weight(df::AbstractDataFrame, weight::Symbol)
	w = convert(Vector{Float64}, df[weight])
	sqrtw = sqrt(w)
end
get_weight(df::AbstractDataFrame, weight::Nothing) = Ones(size(df, 1))





