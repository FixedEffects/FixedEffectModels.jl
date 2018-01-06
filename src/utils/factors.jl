# isnested(a::PooledDataArray, b::PooledDataArray)
# Check whether a factor a is nested within another factor b
#
# This will need to be updated when we move to DataFrames 0.11

function isnested{R <: Integer}(arefs::Vector{R}, brefs::Vector{R})

    #@show arefs
    #@show brefs

    # check size
    if length(arefs) != length(brefs)
        error("isnested(): column vectors need to be of the same size")
    end

    entries_in_b = zeros(R, length(arefs), 1)

    for aind = 1:length(arefs)
        #println("Iter $aind : entries_in_b[arefs[aind]] = $(entries_in_b[arefs[aind]]), entries_in_b[brefs[aind]] = $(entries_in_b[brefs[aind]])")
        if entries_in_b[arefs[aind]] == zero(R)
            # it's a new level, create entry
            entries_in_b[arefs[aind]] = brefs[aind]
        elseif entries_in_b[arefs[aind]] != brefs[aind]
            # not nested: for the same level in a, two different levels in b
            return false
        end
    end

    return true

end
