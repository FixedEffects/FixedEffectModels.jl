
##############################################################################
##
## b is (y - x'b) - (\overline{y} - \overline{x}'b)
##
###############################################################################

# Return vector of vector of estimates
function getfe(fixedeffects::Vector{FixedEffect}, 
               b::Vector{Float64}; 
               maxiter = 10_000_000)

    # construct sparse matrix A
    nobs = length(fixedeffects) * length(b)
    I = Array(Int, nobs)
    J = similar(I)
    V = Array(Float64, nobs)
    start = 0
    idx = 0
    for fe in fixedeffects
        for i in 1:length(fe.refs)
            idx += 1
            I[idx] = i
            J[idx] = start + fe.refs[i]
            V[idx] = fe.interaction[i]
        end
        start += sum(fe.scale .!= 0)
    end

    A = sparse(I, J, V)
   
    # solve Ax = b 
    fevalues0 = A \ b

    # unflatten fevalues0
    fevalues = Vector{Float64}[]
    nstart = 1
    for fe in fixedeffects
        nend = nstart - 1 + sum(fe.scale .!= 0) 
        push!(fevalues, fevalues0[nstart:nend])
        nstart = nend + 1
    end

    # rescale fixed effects 
    interceptindex = find(x -> typeof(x.interaction) <: Ones, fixedeffects)
    if length(interceptindex) >= 2
        components = connectedcomponent(fixedeffects, interceptindex)
        rescale!(fevalues, components, interceptindex)
    end

    return fevalues
end


# Return dataframe of estimates
function getfe(fixedeffects::Vector{FixedEffect},
               b::Vector{Float64}, 
               esample::BitVector; 
               maxiter = 10_000_000)
    
    # return vector of vector of estimates
    fevalues = getfe(fixedeffects, b, maxiter = maxiter)

    # insert matrix into dataframe
    newdf = DataFrame()
    len = length(esample)
    for j in 1:length(fixedeffects)
        name = fixedeffects[j].id
        T = eltype(fixedeffects[j].refs)
        refs = fill(zero(T), len)
        refs[esample] = fixedeffects[j].refs
        newdf[fixedeffects[j].id] = PooledDataArray(RefArray(refs), fevalues[j])
    end
    return newdf
end


##############################################################################
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

function connectedcomponent(fixedeffects::Vector{FixedEffect},
                            interceptindex::Vector{Int})
    # initialize where
    where = initialize_where(fixedeffects)
    refs = initialize_refs(fixedeffects)
    nobs = size(refs, 2)
    visited = fill(false, nobs)
    components = Vector{Set{Int}}[]

    # start
    for i in 1:nobs
        if !visited[i]
            component = Set{Int}[]
            for _ in 1:length(interceptindex)
                push!(component, Set{Int}())
            end
            connectedcomponent!(component, visited, i, refs, where, interceptindex)
            push!(components, component)
        end
    end
    return components
end


function initialize_where(fixedeffects::Vector{FixedEffect})
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


function initialize_refs(fixedeffects::Vector{FixedEffect})
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
function connectedcomponent!(
    component::Vector{Set{Int}}, visited::Vector{Bool},
    i::Integer, refs::Matrix{Int}, where::Vector{Vector{Set{Int}}},
    interceptindex::Vector{Int}
    ) 
    visited[i] = true
    tovisit = Set{Int}()
    # for each fixed effect
    for j in interceptindex
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
            connectedcomponent!(component, visited, k, refs, where, interceptindex)
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

function rescale!(fevalues::Vector{Vector{Float64}}, 
                  components::Vector{Vector{Set{Int}}}, 
                  interceptindex)
    i1 = interceptindex[1]
    adj1 = zero(Float64)
    for component in components
        for i in reverse(interceptindex)
            # demean all fixed effects except the first
            if i != i1
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



