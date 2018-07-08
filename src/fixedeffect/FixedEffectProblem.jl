abstract type FixedEffectProblem end

##############################################################################
##
## A FixedEffectProblem must define three methods:
##
## solve_residuals! 
## solve_coefficients!
## get_fes (accessor)
## 
##############################################################################


##############################################################################
##
## get residuals
##
##############################################################################


function residualize!(X::Union{AbstractVector{Float64}, Matrix{Float64}}, fep::FixedEffectProblem, iterationsv::Vector{Int}, convergedv::Vector{Bool}; kwargs...)
    for j in 1:size(X, 2)
        r, iterations, converged = solve_residuals!(fep, view(X, :, j); kwargs...)
        push!(iterationsv, iterations)
        push!(convergedv, converged)
    end
end

function residualize!(::Array, ::Nothing, 
                      ::Vector{Int}, ::Vector{Bool}; 
                      kwargs...)
    nothing
end


##############################################################################
##
## Get fixed effects
##
## Fixed effects are generally not identified
## We standardize the solution in the following way :
## Mean within connected component of all fixed effects except the first
## is zero
##
## Unique solution with two components, not really with more
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

function getfe!(fep::FixedEffectProblem, b::Vector{Float64}, esample;
                tol::Real = 1e-8, maxiter::Integer = 100_000)
    fev = getfe!(fep, b; tol = tol, maxiter = maxiter)
    fes = get_fes(fep)
    df = DataFrame()
    for j in 1:length(fes)
        df[fes[j].id] = Vector{Union{Float64, Missing}}(missing, length(esample))
        df[esample, fes[j].id] = fev[j][fes[j].refs]
    end
    return df
end


function getfe!(fep::FixedEffectProblem, b::Vector{Float64}; kwargs...)
    # solve Ax = b
    x, iterations, converged = solve_coefficients!(fep, b; kwargs...)
    if !converged 
       warn("getfe did not converge")
    end
    # The solution is generally not unique. Find connected components and scale accordingly
    findintercept = findall(fe -> isa(fe.interaction, Ones), get_fes(fep))
    if length(findintercept) >= 2
        components = connectedcomponent(view(get_fes(fep), findintercept))
        rescale!(x, fep, findintercept, components)
    end
    return x
end


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
            component = Set{Int}[Set{Int}() for fe in fes]
            connectedcomponent!(component, visited, i, refs, where)
            push!(components, component)
        end
    end
    return components
end

function initialize_where(fes::AbstractVector{FixedEffect})
    where = Vector{Set{Int}}[]
    for j in 1:length(fes)
        fe = fes[j]
        wherej = Set{Int}[Set{Int}() for fe in fe.scale]
        for i in 1:length(fe.refs)
            push!(wherej[fe.refs[i]], i)
        end
        push!(where, wherej)
    end
    return where
end

function initialize_refs(fes::AbstractVector{FixedEffect})
    nobs = length(fes[1].refs)
    refs = fill(zero(Int), length(fes), nobs)
    for j in 1:length(fes)
        refs[j, :] = fes[j].refs
    end
    return refs
end

# Breadth-first search
function connectedcomponent!(component::Vector{Set{Int}}, 
    visited::Vector{Bool}, i::Integer, refs::Matrix{Int}, 
    where::Vector{Vector{Set{Int}}}) 
    tovisit = Set{Int}(i)
    while !isempty(tovisit)
        i = pop!(tovisit)
        visited[i] = true
        # for each fixed effect
        for j in 1:size(refs, 1)
            ref = refs[j, i]
            # if category has not been encountered
            if !(ref in component[j])
                # mark category as encountered
                push!(component[j], ref)
                # add other observations with same component in list to visit
                for k in where[j][ref]
                    if !visited[k]
                        push!(tovisit, k)
                    end
                end
            end
        end
    end
end

function rescale!(fev::Vector{Vector{Float64}}, fep::FixedEffectProblem, 
                  findintercept,
                  components::Vector{Vector{Set{Int}}})
    fes = get_fes(fep)
    adj1 = zero(Float64)
    i1 = findintercept[1]
    for component in components
        for i in reverse(findintercept)
            # demean all fixed effects except the first
            if i != 1
                adji = zero(Float64)
                for j in component[i]
                    adji += fev[i][j]
                end
                adji = adji / length(component[i])
                for j in component[i]
                    fev[i][j] -= adji
                end
                adj1 += adji
            else
                # rescale the first fixed effects
                for j in component[i1]
                    fev[i1][j] += adj1
                end
            end
        end
    end
end
