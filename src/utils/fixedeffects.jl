
##############################################################################
##
## isnested
##
##############################################################################

function isnested(fe::FixedEffect, prefs) 
    entries = zeros(eltype(prefs), fe.n)
    for (feref, pref) in zip(fe.refs, prefs)
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

##############################################################################
##
## Number of distinct values
##
##############################################################################

function nunique(fe::FixedEffect)
    out = zeros(Int, fe.n)
    for ref in fe.refs
        out[ref] += 1
    end
    sum(x -> x > 0, out)
end

##############################################################################
##
## Drop Singletons
##
##############################################################################

function drop_singletons!(esample, fe::FixedEffect)
    cache = zeros(Int, fe.n)
    for i in eachindex(esample)
        if esample[i]
            cache[fe.refs[i]] += 1
        end
    end
    for i in eachindex(esample)
        if esample[i] && cache[fe.refs[i]] <= 1
            esample[i] = false
        end
    end
end

##############################################################################
##
## Subet and Make sure Interaction if Vector{Float64} (instead of missing)
##
##############################################################################

# index and convert interaction Vector{Float64,, Missing} to Vector{Missing}
function _subset(fe::FixedEffect, esample)
    interaction = convert(AbstractVector{Float64}, fe.interaction[esample])
    FixedEffect{typeof(fe.refs), typeof(interaction)}(fe.refs[esample], interaction, fe.n)
end
