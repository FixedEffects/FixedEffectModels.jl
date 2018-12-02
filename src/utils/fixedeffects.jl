
##############################################################################
##
## IsNested
##
##############################################################################


function isnested(arefs::AbstractVector{<:Integer}, brefs::AbstractVector{<:Integer}) 
    # check size
    if length(arefs) != length(brefs)
        error("isnested(): column vectors need to be of the same size")
    end
    entries_in_b = zeros(eltype(brefs), length(arefs), 1)
    for aind = 1:length(arefs)
        if entries_in_b[arefs[aind]] == 0
            # it's a new level, create entry
            entries_in_b[arefs[aind]] = brefs[aind]
        elseif entries_in_b[arefs[aind]] != brefs[aind]
            # not nested: for the same level in a, two different levels in b
            return false
        end
    end
    return true
end



##############################################################################
##
## Remove Singletons
##
##############################################################################

function remove_singletons!(esample, fe::FixedEffect)
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
