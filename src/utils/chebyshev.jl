function _chebyshev(x::Vector{Float64}, y::Vector{Float64}, tol::Float64)
	@inbounds for i in 1:length(x)
		if abs(x[i]-y[i]) > tol
			return false
		end
	end
	return true
end

