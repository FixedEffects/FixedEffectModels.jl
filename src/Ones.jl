
##############################################################################
##
## Create light weight type
## 
##############################################################################

type Ones <: AbstractVector{Float64}
    length::Int
end

Base.size(O::Ones) = O.length

Base.getindex(::Ones, i::Int...) = 1.0
Base.unsafe_getindex(::Ones, i::Int...) = 1.0

Base.broadcast!{T}(::Function, ::Array{Float64, T}, ::Array{Float64, T}, ::Ones) = nothing


get_weight(df::AbstractDataFrame, weight::Symbol) = convert(Vector{Float64}, sqrt(df[weight]))
get_weight(df::AbstractDataFrame, ::Void) = Ones(size(df, 1))

function compute_tss(y::Vector{Float64}, hasintercept::Bool, ::Ones)
	if hasintercept
		tss = zero(Float64)
		m = mean(y)::Float64
		@inbounds @simd  for i in 1:length(y)
			tss += abs2((y[i] - m))
		end
	else
		tss = sumabs2(y)
	end
	return tss
end

function compute_tss(y::Vector{Float64}, hasintercept::Bool, sqrtw::Vector{Float64})
	if hasintercept
		m = (mean(y) / sum(sqrtw) * length(y))::Float64
		tss = zero(Float64)
		@inbounds @simd  for i in 1:length(y)
			tss += abs2(y[i] - sqrtw[i] * m)
		end
	else
		tss = sumabs2(y)
	end
	return tss
end
