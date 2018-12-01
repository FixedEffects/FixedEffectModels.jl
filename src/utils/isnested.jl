function isnested(arefs::Vector{R}, brefs::Vector{T}) where {R <: Integer, T <: Integer}
    # check size
    if length(arefs) != length(brefs)
        error("isnested(): column vectors need to be of the same size")
    end
    entries_in_b = zeros(T, length(arefs), 1)
    for aind = 1:length(arefs)
        if entries_in_b[arefs[aind]] == zero(T)
            # it's a new level, create entry
            entries_in_b[arefs[aind]] = brefs[aind]
        elseif entries_in_b[arefs[aind]] != brefs[aind]
            # not nested: for the same level in a, two different levels in b
            return false
        end
    end
    return true
end



