
##############################################################################
##
## Create light weight type
## 
##############################################################################

type Ones <: AbstractVector{Float64}
    length::Int
end
Base.length(O::Ones) = O.length
Base.size(O::Ones) = O.length
convert{T}(::Type{Vector{T}}, o::Ones) = ones(T, length(o))
Base.similar(o::Ones) = Ones(length(o))
Base.copy(o::Ones) = Ones(length(o))
Base.deepcopy(o::Ones) = Ones(length(o))

#indexing
Base.linearindexing(::Type{Ones}) = Base.LinearFast()
@inline Base.getindex(::Ones, i::Int...) = 1.0
@inline Base.unsafe_getindex(::Ones, i::Int...) = 1.0

# implement map
# implement broadcast
Base.broadcast!{T}(::Function, x::Array{Float64, T}, ::Array{Float64, T}, ::Ones) = x


function get_weight(df::AbstractDataFrame, esample::BitVector, weight::Symbol) 
	out = df[esample, weight]
	# there are no NA in it. DataVector to Vector
	out = convert(Vector{Float64}, out)
	map!(sqrt, out, out)
	return out
end
get_weight(df::AbstractDataFrame, esample::BitVector, ::Void) = Ones(sum(esample))

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
