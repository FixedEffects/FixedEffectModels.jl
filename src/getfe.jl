
##############################################################################
##
## Get coefficients for high dimensional fixed effects
## 
###############################################################################


# Return vector of vector of estimates
function getfe(pfe::FixedEffectProblem, b::Vector{Float64};  maxiter = 100_000)
    
    # solve Ax = b
    fes = pfe.m._
    x = zeros(size(pfe.m, 2))
    iterations, converged = cgls!(x, b, pfe, tol = 1e-10, maxiter = maxiter)
    if !converged 
       warn("did not converge")
    end

    # unflatten x -> fevalues
    fevalues = Vector{Float64}[]
    idx = 0
    for i in 1:length(fes)
        fe = fes[i]
        push!(fevalues, Array(Float64, length(fe.scale)))
        for j in 1:length(fe.scale)
            idx += 1
            fevalues[i][j] = x[idx] * fe.scale[j]
        end
    end        

    # find connected components and scale accordingly
    findintercept = find(x -> typeof(x.interaction) <: Ones, fes)
    if length(findintercept) >= 2
        if VERSION >= v"0.4.0-dev+6521" 
            components = connectedcomponent(sub(fes, findintercept))
        else
            components = connectedcomponent(fes[findintercept])
        end
        rescale!(fevalues, findintercept, components)
    end

    return fevalues
end


# Convert estimates to dataframes 
function DataFrame(fes::Vector{FixedEffect}, fevalues, esample::BitVector; maxiter = 100_000)
    newdf = DataFrame()
    len = length(esample)
    for j in 1:length(fes)
        name = fes[j].id
        T = eltype(fes[j].refs)
        refs = fill(zero(T), len)
        refs[esample] = fes[j].refs
        newdf[fes[j].id] = PooledDataArray(RefArray(refs), fevalues[j])
    end
    return newdf
end

function getfe(pfe::FixedEffectProblem, b::Vector{Float64}, 
               esample::BitVector; maxiter = 100_000)
    fevalues = getfe(pfe, b, maxiter = maxiter)
    DataFrame(pfe.m._, fevalues, esample)
end


##############################################################################
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

function connectedcomponent(fixedeffects::AbstractVector{FixedEffect})

    # initialize
    where = initialize_where(fixedeffects)
    refs = initialize_refs(fixedeffects)
    nobs = size(refs, 2)
    visited = fill(false, nobs)
    components = Vector{Set{Int}}[]

    # start
    for i in 1:nobs
        if !visited[i]
            component = Set{Int}[]
            for _ in 1:length(fixedeffects)
                push!(component, Set{Int}())
            end
            connectedcomponent!(component, visited, i, refs, where)
            push!(components, component)
        end
    end
    return components
end

function initialize_where(fixedeffects::AbstractVector{FixedEffect})
    where = Vector{Set{Int}}[]
    for j in 1:length(fixedeffects)
        push!(where, Set{Int}[])
        fe = fixedeffects[j]
        for _ in 1:length(fe.scale)
            push!(where[j], Set{Int}())
        end
        @inbounds for i in 1:length(fe.refs)
            push!(where[j][fe.refs[i]], i)
        end
    end
    return where
end

function initialize_refs(fixedeffects::AbstractVector{FixedEffect})
    nobs = length(fixedeffects[1].refs)
    refs = fill(zero(Int), length(fixedeffects), nobs)
    for j in 1:length(fixedeffects)
        ref = fixedeffects[j].refs
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

function rescale!(fevalues::AbstractVector{Vector{Float64}}, 
                    findintercept,
                  components::Vector{Vector{Set{Int}}})
    adj1 = zero(Float64)
    i1 = findintercept[1]
    for component in components
        for i in reverse(findintercept)
            # demean all fixed effects except the first
            if i != 1
                adji = zero(Float64)
                for j in component[i]
                    adji += fevalues[i][j]
                end
                adji = adji / length(component[i])
                for j in component[i]
                    fevalues[i][j] -= adji
                end
                adj1 += adji
            else
                # rescale the first fixed effects
                for j in component[i1]
                    fevalues[i1][j] += adj1
                end
            end
        end
    end
end



