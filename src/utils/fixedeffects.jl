
##############################################################################
##
## IsNested
##
##############################################################################

function isnested(fe::FixedEffect, p::CategoricalVector) 
    entries_in_p = Dict{eltype(fe.refs), eltype(p.refs)}()
    sizehint!(entries_in_p, fe.n)
    for (feref, pref) in zip(fe.refs, p.refs)
        x = get(entries_in_p, feref, 0)
        if x == 0
            # it's a new level, create entry
            entries_in_p[feref] = pref
        elseif x != pref
            # not nested: for the same level in a, two different levels in b
            return false
        end
    end
    return true
end

##############################################################################
##
## Drop Singletons
##
##############################################################################

function drop_singletons!(esample, fe::FixedEffect)
    cache = zeros(Int, fe.n)
    for i in 1:length(esample)
        if esample[i]
            cache[fe.refs[i]] += 1
        end
    end
    for i in 1:length(esample)
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
    FixedEffect{typeof(fe.refs), Vector{Float64}}(fe.refs[esample], convert(Vector{Float64}, view(fe.interaction, esample)), fe.n)
end
