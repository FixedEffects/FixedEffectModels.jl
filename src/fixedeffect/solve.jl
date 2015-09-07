##############################################################################
##
## get residuals
##
##############################################################################

function residualize!(x::AbstractVector{Float64}, pfe::FixedEffectProblem, 
                      iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    iterations, converged = cgls!(nothing, x, pfe;  maxiter = maxiter, tol = tol)
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

function residualize!(::Array, ::Nothing, 
                      ::Vector{Int}, ::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end



##############################################################################
##
## get fixed effect estimates
## 
###############################################################################

function solvefe!(pfe::FixedEffectProblem, b::Vector{Float64};  maxiter = 100_000)
    
    # solve Ax = b
    fes = pfe.m._
    x = zeros(size(pfe.m, 2))
    iterations, converged = cgls!(x, b, pfe, tol = 1e-10, maxiter = maxiter)
    if !converged 
       warn("did not converge")
    end
    copy!(pfe.m, x) 

    # The solution is generally not unique. Find connected components and scale accordingly
    findintercept = find(fe -> isa(fe.interaction, Ones), fes)
    if length(findintercept) >= 2
        if VERSION >= v"0.4.0-dev+6521" 
            components = connectedcomponent(sub(fes, findintercept))
        else
            components = connectedcomponent(fes[findintercept])
        end
        rescale!(pfe, findintercept, components)
    end

    return pfe
end

# Convert estimates to dataframes 
function DataFrame(pfe::FixedEffectProblem, esample::BitVector)
    fes = pfe.m._
    newdf = DataFrame()
    len = length(esample)
    for j in 1:length(fes)
        name = fes[j].id
        T = eltype(fes[j].refs)
        refs = fill(zero(T), len)
        refs[esample] = fes[j].refs
        newdf[fes[j].id] = PooledDataArray(RefArray(refs), fes[j].value)
    end
    return newdf
end

function solvefe!(pfe::FixedEffectProblem, b::Vector{Float64}, 
               esample::BitVector; maxiter = 100_000)
    solvefe!(pfe, b, maxiter = maxiter)
    DataFrame(pfe, esample)
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

function rescale!(pfe::FixedEffectProblem, 
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
                    adji += fes[i].value[j]
                end
                adji = adji / length(component[i])
                for j in component[i]
                    fes[i].value[j] -= adji
                end
                adj1 += adji
            else
                # rescale the first fixed effects
                for j in component[i1]
                    fes[i1].value[j] += adj1
                end
            end
        end
    end
end
