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
Base.size(O::Ones) = (O.length,)

Base.similar{T}(o::Ones{T}) = Ones{T}(length(o))
Base.copy{T}(o::Ones{T}) = Ones{T}(length(o))
Base.deepcopy{T}(o::Ones{T}) = Ones{T}(length(o))
Base.diagm{T}(o::Ones{T}, args...) = eye(T, O.length, args...)

Base.sum(O::Ones) = O.length
Base.convert{T}(::Type{Vector{T}}, o::Ones) = ones(T, length(o))
Base.collect{T}(o::Ones{T}) = ones(T, length(o))


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
