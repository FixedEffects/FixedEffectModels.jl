##############################################################################
##
## Drop Singletons
##
##############################################################################

function drop_singletons!(esample, fes::Vector{<:FixedEffect}, nthreads)
# Main._esample = copy(esample); Main._fes=copy(fes); Main._nthreads=nthreads
    nsingletons = 0
    dirtypasses = length(fes)
    
    bounds = ceil.(Int, (0:nthreads) * (length(esample) / nthreads))
    chunks = [bounds[t]+1:bounds[t+1] for t ∈ 1:nthreads]
  
    caches = [Vector{UInt8}(undef, fes[i].n) for i ∈ eachindex(fes)]
    for (fe,cache) ∈ Iterators.cycle(zip(fes,caches))  # if 1 set of FE, only need 1 pass
        n = drop_singletons!(esample, fe, cache, chunks)
        if iszero(n)
            dirtypasses -= 1
        else
            nsingletons += n
            dirtypasses = length(fes)  # restart counter
        end
        dirtypasses <= 1 && break  # done if there are N FE groups and at least last N-1 have been clean (includes case N=1)
    end
  
    nsingletons
end

function drop_singletons!(esample, fe::FixedEffect, cache, chunks)::Int
# Main._cache = cache; Main._chunks = copy(chunks)
    refs = fe.refs
      
    fill!(cache, zero(UInt8))
    @inbounds for i ∈ eachindex(esample,refs)  # count obs in each FE group
        if esample[i]
            cache[refs[i]] = min(0x02, cache[refs[i]] + 0x01)  # stop counting obs in a group at 2, to keep counters in 8-bit integers
        end
    end

    tasks = map(chunks) do chunk  # drop newly found singletons
        Threads.@spawn begin
            n = 0
            @inbounds for i ∈ chunk
                if esample[i] && isone(cache[refs[i]])
                    esample[i] = false
                    n += 1
                end
            end
            n
        end
    end

    sum(fetch.(tasks))
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
