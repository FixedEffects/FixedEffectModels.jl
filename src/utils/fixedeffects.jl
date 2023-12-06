##############################################################################
##
## Drop Singletons
##
##############################################################################
function drop_singletons!(esample, fes::Vector{<:FixedEffect))
    ns = Int[]
    for fe in Iterators.cycle(fes)
        # break loop if number of singletons did not change since the last time fe was iterated on
        if length(ns) >= length(fes) && sum(view(ns, end-length(fes)+1, end)) == ns[end-length(fes)+1]
            break
        end
        push!(ns, drop_singletons!(esample, fe))
    end
    return sum(ns)
end




function drop_singletons!(esample, fe::FixedEffect)
    n = 0
    cache = zeros(Int, fe.n)
    @inbounds for i in eachindex(esample)
        if esample[i]
            cache[fe.refs[i]] += 1
        end
    end
    @inbounds for i in eachindex(esample)
        if esample[i] && (cache[fe.refs[i]] == 1)
            esample[i] = false
            n += 1
        end
    end
    return n
end

##############################################################################
##
## Number of distinct values (only ever call when fe without missing values)
## 
##############################################################################

function nunique(fe::FixedEffect)
    out = zeros(Int, fe.n)
    @inbounds @simd for ref in fe.refs
        out[ref] += 1
    end
    sum(>(0), out)
end


##############################################################################
##
## isnested
##
##############################################################################

function isnested(fe::FixedEffect, prefs) 
    entries = zeros(eltype(prefs), fe.n)
    @inbounds for (feref, pref) in zip(fe.refs, prefs)
        if entries[feref] == 0
            # it's a new level, create entry
            entries[feref] = pref
        elseif entries[feref] != pref
            # not nested: for the same level in a, two different levels in b
            return false
        end
    end
    return true
end
