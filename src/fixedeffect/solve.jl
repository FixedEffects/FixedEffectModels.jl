##############################################################################
##
## FixedEffectProblem is a wrapper around a FixedEffectMatrix 
## with some storage arrays used when solving (A'A)X = A'y 
##
##############################################################################

type FixedEffectProblem
    m::FixedEffectMatrix
    fev::NTuple{6, FixedEffectVector}
    b::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect})
    m = FixedEffectMatrix(fes)
    fev = (FixedEffectVector(fes), FixedEffectVector(fes), FixedEffectVector(fes), 
            FixedEffectVector(fes), FixedEffectVector(fes), FixedEffectVector(fes))
    b = Array(Float64, length(fes[1].refs))
    return FixedEffectProblem(m, NTuple(fev), b)
end

##############################################################################
##
## get residuals
##
##############################################################################

function residualize!(y::AbstractVector{Float64}, fep::FixedEffectProblem, 
                      iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    x, normalization, tmp, v, h, hbar = fep.fev
    fill!(x, zero(Float64))
    b = fep.b
    copy!(b, y)
    lls = LinearLeastSquares(x, b, fep.m)
    alls = LinearLeastSquaresAllocated(lls, NTuple((normalization, tmp, v, h, hbar))) 
    result = optimize!(alls, atol = tol, btol = tol, iterations = maxiter)
    A_mul_B!(-1.0, fep.m, x, 1.0, y)
    push!(iterationsv, result.iterations)
    push!(convergedv, result.converged)
end

function residualize!(X::Matrix{Float64}, fep::FixedEffectProblem, 
                      iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        residualize!(slice(X, :, j), fep, iterationsv, convergedv, maxiter = maxiter, tol = tol)
    end
end

function residualize!(::Array, ::Void, 
                      ::Vector{Int}, ::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end

##############################################################################
##
## get fixed effects
## 
###############################################################################

function getfe!(fep::FixedEffectProblem, b::Vector{Float64};  
                tol::Real = 1e-8, maxiter::Integer = 100_000) 

    # solve Ax = b
    x, normalization, tmp, v, h, hbar = fep.fev
    fill!(x, zero(Float64))
    lls = LinearLeastSquares(x, b, fep.m)
    alls = LinearLeastSquaresAllocated(lls, NTuple((normalization, tmp, v, h, hbar)))
    result = optimize!(alls, atol = tol, btol = tol, iterations = maxiter)
    if !result.converged 
       warn("getfe did not converge")
    end

    # The solution is generally not unique. Find connected components and scale accordingly
    findintercept = find(fe -> isa(fe.interaction, Ones), fep.m._)
    if length(findintercept) >= 2
        components = connectedcomponent(sub(fep.m._, findintercept))
        rescale!(x, fep, findintercept, components)
    end

    return x
end

# Convert estimates to dataframes 
function DataFrame(fev::FixedEffectVector, fep::FixedEffectProblem, esample::BitVector)
    fes = fep.m._
    newdf = DataFrame()
    len = length(esample)
    for j in 1:length(fes)
        name = fes[j].id
        T = eltype(fes[j].refs)
        refs = fill(zero(T), len)
        refs[esample] = fes[j].refs
        newdf[fes[j].id] = PooledDataArray(RefArray(refs), fev._[j])
    end
    return newdf
end

function getfe!(fep::FixedEffectProblem, b::Vector{Float64},esample::BitVector;
                tol::Real = 1e-8, maxiter::Integer = 100_000)
    fev = getfe!(fep, b; tol = tol, maxiter = maxiter)
    return DataFrame(fev, fep, esample)
end


##############################################################################
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

function connectedcomponent(fes::AbstractVector{FixedEffect})
    # initialize
    where = initialize_where(fes)
    refs = initialize_refs(fes)
    nobs = size(refs, 2)
    visited = fill(false, nobs)
    components = Vector{Set{Int}}[]

    # start
    for i in 1:nobs
        if !visited[i]
            component = Set{Int}[]
            for _ in 1:length(fes)
                push!(component, Set{Int}())
            end
            connectedcomponent!(component, visited, i, refs, where)
            push!(components, component)
        end
    end
    return components
end

function initialize_where(fes::AbstractVector{FixedEffect})
    where = Vector{Set{Int}}[]
    for j in 1:length(fes)
        push!(where, Set{Int}[])
        fe = fes[j]
        for _ in 1:length(fe.scale)
            push!(where[j], Set{Int}())
        end
        @inbounds for i in 1:length(fe.refs)
            push!(where[j][fe.refs[i]], i)
        end
    end
    return where
end

function initialize_refs(fes::AbstractVector{FixedEffect})
    nobs = length(fes[1].refs)
    refs = fill(zero(Int), length(fes), nobs)
    for j in 1:length(fes)
        ref = fes[j].refs
        for i in 1:length(ref)
            refs[j, i] = ref[i]
        end
    end
    return refs
end

# Breadth-first search
function connectedcomponent!(component::Vector{Set{Int}}, 
    visited::Vector{Bool}, i::Integer, refs::Matrix{Int}, 
    where::Vector{Vector{Set{Int}}}) 
    visited[i] = true
    tovisit = Set{Int}()
    # for each fixed effect
    for j in 1:size(refs, 1)
        ref = refs[j, i]
        # if category has not been encountered
        if !(ref in component[j])
            # mark category as encountered
            push!(component[j], ref)
            # add other observations with same component in list to visit
            for k in where[j][ref]
                push!(tovisit, k)
            end
        end
    end
    for k in tovisit
        if k != i
            connectedcomponent!(component, visited, k, refs, where)
        end
    end
end

##############################################################################
##
## rescale fixed effect to make solution unique (at least in case of 2 fixed effects)
## normalization: for each factor except the first one, mean within each component is 0 
## Unique solution with two components, not really with more
##
###############################################################################

function rescale!(fev::FixedEffectVector, fep::FixedEffectProblem, 
                  findintercept,
                  components::Vector{Vector{Set{Int}}})
    fes = fep.m._
    adj1 = zero(Float64)
    i1 = findintercept[1]
    for component in components
        for i in reverse(findintercept)
            # demean all fixed effects except the first
            if i != 1
                adji = zero(Float64)
                for j in component[i]
                    adji += fev._[i][j]
                end
                adji = adji / length(component[i])
                for j in component[i]
                    fev._[i][j] -= adji
                end
                adj1 += adji
            else
                # rescale the first fixed effects
                for j in component[i1]
                    fev._[i1][j] += adj1
                end
            end
        end
    end
end
