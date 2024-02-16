##############################################################################
##
## Drop Singletons
##
##############################################################################

function drop_singletons!(esample, fes::Vector{<:FixedEffect}, nthreads)
    nsingletons = 0
    ncleanpasses = 0
    caches = [Vector{UInt8}(undef, fes[i].n) for i in eachindex(fes)]
    for (fe, cache) in Iterators.cycle(zip(fes,caches))
        n = drop_singletons!(esample, fe, cache)
        nsingletons += n
        if n > 0
            # found singletons, reset the counter
            ncleanpasses = 0
        else
            # otherwise, increment counter
            ncleanpasses += 1
        end
        # if the last N-1 passes have not found singletons (where N is number of FE groups), break the loop
        ncleanpasses >= length(fes) - 1 && break  
    end
    return nsingletons
end

function drop_singletons!(esample, fe::FixedEffect, cache)
    refs = fe.refs
    fill!(cache, 0)
    @inbounds for i in eachindex(esample, refs)  # count obs in each FE group
        if esample[i]
            # no need to keep counting obs after 2 (counters are 8-bit integers)
            cache[refs[i]] = min(0x02, cache[refs[i]] + 0x01)  
        end
    end
    n = 0
    @inbounds for i in eachindex(esample, refs)
        if esample[i] && cache[refs[i]] == 0x01
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
