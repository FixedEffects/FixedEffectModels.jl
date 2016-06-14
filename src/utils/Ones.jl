
##############################################################################
##
## Create light weight type
## 
##############################################################################

type Ones{T} <: AbstractVector{T}
    length::Int
end

# constructuor mimic ones syntax
Ones(T::Type, v::Int) = Ones{T}(v)
Ones(v::Integer) = Ones{Float64}(v)
Ones{T}(v::AbstractVector{T}) = Ones{T}(length(v))

#indexing
Base.linearindexing(::Type{Ones}) = Base.LinearFast()
@inline Base.getindex{T}(::Ones{T}, i::Int...) = one(T)
@inline Base.unsafe_getindex{T}(::Ones{T}, i::Int...) = one(T)
Base.eltype{T}(o::Ones{T}) = T
Base.length(O::Ones) = O.length
Base.size(O::Ones) = O.length

Base.similar{T}(o::Ones{T}) = Ones{T}(length(o))
Base.copy{T}(o::Ones{T}) = Ones{T}(length(o))
Base.deepcopy{T}(o::Ones{T}) = Ones{T}(length(o))
Base.diagm{T}(o::Ones{T}, args...) = eye(T, O.length, args...)

Base.sum(O::Ones) = O.length
Base.convert{T}(::Type{Vector{T}}, o::Ones) = ones(T, length(o))
Base.collect{T}(o::Ones{T}) = ones(T, length(o))
if VERSION > v"0.5.0-"
	Base.shape{T}(o::Ones{T}) = (length(o),)
end

# implement broadcast
## solve ambiguity
for t in (BitArray, DataArray, PooledDataArray)
	@eval begin
		function Base.broadcast!(op::Function, A::$t, o::Ones)
			invoke(broadcast!, (Any, Any, Ones), op, A, o)
		end
	end
end

function Base.broadcast!(op::Function, A::Any, o::Ones) 
	if op == *
		A
	else 
		invoke(broadcast!, (Any, typeof(A), AbstractVector), op, A, o)
	end
end


##############################################################################
##
## Use
## 
##############################################################################

function get_weight(df::AbstractDataFrame, esample::BitVector, weight::Symbol) 
	out = df[esample, weight]
	# there are no NA in it. DataVector to Vector
	out = convert(Vector{Float64}, out)
	map!(sqrt, out, out)
	return out
end
get_weight(df::AbstractDataFrame, esample::BitVector, ::Void) = Ones{Float64}(sum(esample))

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
