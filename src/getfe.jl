
##############################################################################
##
## Fixed effects solve Ax = b
## b is (y - x'b) - (\overline{y} - \overline{x}'b)
## TODO 1 : use sparse matrix inversion from Base? 
## TODO 2 : Use only unique rows?
###############################################################################


# Return vector of vector of estimates
function getfe(fixedeffects::Vector{FixedEffect}, 
               b::Vector{Float64}; 
               maxiter = 10_000_000)
    ## initialize data structures
    interceptindex = find(fe->typeof(fe.interaction) <: Ones, fixedeffects)
    fevalues, where, refs, A = initialize(fixedeffects)
    
    # solve Ax = b by kaczmarz algorithm
    converged = kaczmarz!(fevalues, b, refs, A, maxiter)

    # rescale fixed effects 
    if length(interceptindex) >= 2
        components = connectedcomponent(refs, where, interceptindex)
        rescale!(fevalues, components, interceptindex)
    end

    if !converged
        warn("Estimation of fixed effects did not converged")
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
## Initialize structures
## Denote refs(j) the reference vector of the jth factor
## where[j][i] is a set that stores indices k with refs(j)[k] = i
## fevalues[j][i] is a Float64 that stores the fixed effect estimate with refs = i
## refs[j, i] is a refs(j)[i]
##
##############################################################################

function initialize(fixedeffects::Vector{FixedEffect})
    nobs = length(fixedeffects[1].refs)
    fevalues = Array(Vector{Float64}, length(fixedeffects)) 
    where = Array(Vector{Set{Int}}, length(fixedeffects))
    refs = fill(zero(Int), length(fixedeffects), nobs)
    A = fill(one(Float64), length(fixedeffects),  nobs)
    j = 0
    for f in fixedeffects
        j += 1
        initialize!(j, f, fevalues, where, refs, A)
    end
    return fevalues, where, refs, A
end

function initialize!{R, W, I}(
    j::Int, f::FixedEffect{R, W, I}, fevalues::Vector{Vector{Float64}}, 
    where::Vector{Vector{Set{Int}}}, refs::Matrix{Int}, A::Matrix{Float64}
    )
    fevalues[j] = fill(zero(Float64), length(f.scale))
    where[j] = Set{Int}[]
    # fill would create a reference to the same object
    for i in 1:length(f.scale)
        push!(where[j], Set{Int}())
    end
    @inbounds for i in 1:length(f.refs)
        refi = f.refs[i]
        refs[j, i] = refi
        A[j, i] = f.interaction[i]
        push!(where[j][refi], i)
    end
end

##############################################################################
##
## Randomized Kaczmarz algorithm 
## https://en.wikipedia.org/wiki/Kaczmarz_method
##
##############################################################################


function kaczmarz!(
    fevalues::Vector{Vector{Float64}}, b::Vector{Float64},
    refs::Matrix{Int}, A::Matrix{Float64}, maxiter::Integer
    )
    # precompute invnorm = 1/norm[i] since division costly
    norm = fill(zero(Float64), size(A, 2))
    invnorm = fill(zero(Float64), size(A, 2))
    @inbounds for i in 1:size(A, 2)
        out = zero(Float64)
        for j in 1:size(A, 1)
            out += abs2(A[j, i])
        end
        norm[i] = out
        invnorm[i] = 1 / out
    end

    # sampling distribution for rows
    # Needell, Srebro, Ward (2015)
    dist = 1:length(b)
    return kaczmarz!(fevalues, b, refs, A, maxiter, dist, invnorm)
end

function kaczmarz!(
    fevalues::Vector{Vector{Float64}},b::Vector{Float64},
    refs::Matrix{Int}, A::Matrix{Float64}, maxiter::Integer,
    dist, invnorm::Vector{Float64})
    len_fe = length(fevalues)
    len_b = length(b)
    iter = 0
    @inbounds while iter < maxiter 
        iter += 1
        for _ in 1:len_b
            currenterror = zero(Float64)
            inner_iter = zero(Int)
                
            # draw a row
            i = rand(dist)
            
            # compute update = (b_i - <x_k, a_i>)/||a_i||^2
            numerator = b[i]
            for j in 1:len_fe
                numerator -= fevalues[j][refs[j, i]] * A[j, i]
            end
            update = numerator * invnorm[i]

            # update x_k
            for j in 1:len_fe   
                fevalues[j][refs[j, i]] +=  update * A[j, i]
            end
        end
        # check maxabs(Ax-b) 
        if maxabs(fevalues, b, refs, A, 1e-4)
            return true
        end
    end
    return false
end


function maxabs(
    fevalues::Vector{Vector{Float64}}, b::Vector{Float64},
    refs::Matrix{Int}, A::Matrix{Float64}, tol::Float64)
    len_fe = length(fevalues)
    @inbounds for i in 1:length(b)
        current = b[i]
        for j in 1:len_fe
            current -= fevalues[j][refs[j, i]] * A[j, i]
        end
        if abs(current) > tol
            return false
        end
    end
    return true
end


##############################################################################
##
## Connected component : Breadth-first search
## components is an array of component
## A component is an array of set (length is number of values taken)
##
##############################################################################

function connectedcomponent(refs::Matrix{Int},
                            where::Vector{Vector{Set{Int}}},
                            interceptindex::Vector{Int})
    nobs = size(refs, 2)
    visited = fill(false, nobs)
    components = Vector{Set{Int}}[]
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



