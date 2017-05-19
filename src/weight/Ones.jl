##############################################################################
##
## Create light weight type
##
##############################################################################

struct Ones{T} <: AbstractVector{T}
    length::Int
end

# constructuor mimic ones syntax
Ones(T::Type, v::Int) = Ones{T}(v)
Ones(v::Integer) = Ones{Float64}(v)
Ones(v::AbstractVector{T}) where {T} = Ones{T}(length(v))

#indexing
Base.IndexStyle(::Type{Ones}) = Base.LinearFast()
@inline Base.getindex(::Ones{T}, i::Int...) where {T} = one(T)
@inline Base.unsafe_getindex(::Ones{T}, i::Int...) where {T} = one(T)
Base.eltype(o::Ones{T}) where {T} = T
Base.length(O::Ones) = O.length
Base.size(O::Ones) = (O.length,)

Base.similar(o::Ones{T}) where {T} = Ones{T}(length(o))
Base.copy(o::Ones{T}) where {T} = Ones{T}(length(o))
Base.deepcopy(o::Ones{T}) where {T} = Ones{T}(length(o))
Base.diagm(o::Ones{T}, args...) where {T} = eye(T, O.length, args...)

Base.sum(O::Ones) = O.length
Base.convert(::Type{Vector{T}}, o::Ones) where {T} = ones(T, length(o))
Base.collect(o::Ones{T}) where {T} = ones(T, length(o))


# implement broadcast
## solve ambiguity
for t in (BitArray, DataArray, PooledDataArray)
	@eval begin
		function Base.broadcast!(op::Function, A::$t, o::Ones)
			invoke(broadcast!, Tuple{Any,Any,Ones}, op, A, o)
		end
	end
end

function Base.broadcast!(op::Function, A::Any, o::Ones)
	if op == *
		A
	else
		invoke(broadcast!, Tuple{Any,typeof(A),AbstractVector}, op, A, o)
	end
end
