##############################################################################
##
## FixedEffectProblem is a wrapper around a FixedEffectMatrix 
## with some storage arrays used when solving (A'A)X = A'y 
##
##############################################################################

type FixedEffectProblem
    m::FixedEffectMatrix
    x::FixedEffectVector
    v::FixedEffectVector
    h::FixedEffectVector
    hbar::FixedEffectVector
    u::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect})
    m = FixedEffectMatrix(fes)
    x = FixedEffectVector(fes)
    v = FixedEffectVector(fes)
    h = FixedEffectVector(fes)
    hbar = FixedEffectVector(fes)
    u = Array(Float64, length(fes[1].refs))

    FixedEffectProblem(m, x, v, h, hbar, u)
end

function lsmr!(x, r, pfe::FixedEffectProblem; tol::Real=1e-8, maxiter::Integer=1000)
    lsmr!(x, r, pfe.m, pfe.u, pfe.v, pfe.h, pfe.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
end

function lsmr!(::Void, r, pfe::FixedEffectProblem; tol::Real=1e-8, maxiter::Integer=1000)
    fill!(pfe.x, zero(Float64))
    lsmr!(pfe.x, r, pfe.m, pfe.u, pfe.v, pfe.h, pfe.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
end

##############################################################################
##
## get residuals
##
##############################################################################

function residualize!(x::AbstractVector{Float64}, pfe::FixedEffectProblem, 
                      iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    iterations, converged = lsmr!(nothing, x, pfe;  maxiter = maxiter, tol = tol)
    push!(iterationsv, iterations)
    push!(convergedv, converged)
end

function residualize!(X::Matrix{Float64}, pfe::FixedEffectProblem, 
                      iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        residualize!(slice(X, :, j), pfe, iterationsv, convergedv, maxiter = maxiter, tol = tol)
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

function getfe!(pfe::FixedEffectProblem, b::Vector{Float64};  
                tol::Real = 1e-8, maxiter::Integer = 100_000)
    
    # solve Ax = b
    fes = pfe.m._
    vfe = FixedEffectVector(fes)
    fill!(vfe, zero(Float64))
    iterations, converged = lsmr!(vfe, b, pfe; tol = tol, maxiter = maxiter)
    if !converged 
       warn("getfe did not converge")
    end
    for i in 1:length(vfe)
        broadcast!(*, vfe[i], vfe[i], pfe.m._[i].scale)
    end

    # The solution is generally not unique. Find connected components and scale accordingly
    findintercept = find(fe -> isa(fe.interaction, Ones), fes)
    if length(findintercept) >= 2
        components = connectedcomponent(sub(fes, findintercept))
        rescale!(vfe, pfe, findintercept, components)
    end

    return vfe
end

# Convert estimates to dataframes 
function DataFrame(vfe::FixedEffectVector, pfe::FixedEffectProblem, esample::BitVector)
    fes = pfe.m._
    newdf = DataFrame()
    len = length(esample)
    for j in 1:length(fes)
        name = fes[j].id
        T = eltype(fes[j].refs)
        refs = fill(zero(T), len)
        refs[esample] = fes[j].refs
        newdf[fes[j].id] = PooledDataArray(RefArray(refs), vfe[j])
    end
    return newdf
end

function getfe!(pfe::FixedEffectProblem, b::Vector{Float64},esample::BitVector;
                tol::Real = 1e-8, maxiter::Integer = 100_000)
    vfe = getfe!(pfe, b; tol = tol, maxiter = maxiter)
    DataFrame(vfe, pfe, esample)
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

function rescale!(vfe::FixedEffectVector, pfe::FixedEffectProblem, 
                  findintercept,
                  components::Vector{Vector{Set{Int}}})
    fes = pfe.m._
    adj1 = zero(Float64)
    i1 = findintercept[1]
    for component in components
        for i in reverse(findintercept)
            # demean all fixed effects except the first
            if i != 1
                adji = zero(Float64)
                for j in component[i]
                    adji += vfe[i][j]
                end
                adji = adji / length(component[i])
                for j in component[i]
                    vfe[i][j] -= adji
                end
                adj1 += adji
            else
                # rescale the first fixed effects
                for j in component[i1]
                    vfe[i1][j] += adj1
                end
            end
        end
    end
end
