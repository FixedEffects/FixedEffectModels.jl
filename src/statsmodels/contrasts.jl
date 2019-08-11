# Specify contrasts for coding categorical data in model matrix. Contrasts types
# are a subtype of AbstractContrasts. ContrastsMatrix types hold a contrast
# matrix, levels, and term names and provide the interface for creating model
# matrix columns and coefficient names.
#
# Contrasts types themselves can be instantiated to provide containers for
# contrast settings (currently, just the base level).
#
# ModelFrame will hold a Dict{Symbol, ContrastsMatrix} that maps column
# names to contrasts.
#
# ModelMatrix will check this dict when evaluating terms, falling back to a
# default for any categorical data without a specified contrast.


abstract type AbstractContrasts end

# Contrasts + Levels (usually from data) = ContrastsMatrix
mutable struct ContrastsMatrix{C <: AbstractContrasts, T}
    matrix::Matrix{Float64}
    termnames::Vector{T}
    levels::Vector{T}
    contrasts::C
end

# only check equality of matrix, termnames, and levels, and that the type is the
# same for the contrasts (values are irrelevant).  This ensures that the two
# will behave identically in creating modelmatrix columns
Base.:(==)(a::ContrastsMatrix{C,T}, b::ContrastsMatrix{C,T}) where {C<:AbstractContrasts,T} =
    a.matrix == b.matrix &&
    a.termnames == b.termnames &&
    a.levels == b.levels

Base.hash(a::ContrastsMatrix{C}, h::UInt) where {C} =
    hash(C, hash(a.matrix, hash(a.termnames, hash(a.levels, h))))

function ContrastsMatrix(contrasts::AbstractContrasts, levels::AbstractVector)

    # if levels are defined on contrasts, use those, validating that they line up.
    # what does that mean? either:
    #
    # 1. contrasts.levels == levels (best case)
    # 2. data levels missing from contrast: would generate empty/undefined rows.
    #    better to filter data frame first
    # 3. contrast levels missing from data: would have empty columns, generate a
    #    rank-deficient model matrix.
    c_levels = something(contrasts.levels, levels)
    if eltype(c_levels) != eltype(levels)
        throw(ArgumentError("mismatching levels types: got $(eltype(levels)), expected " *
                            "$(eltype(c_levels)) based on contrasts levels."))
    end
    mismatched_levels = symdiff(c_levels, levels)
    if !isempty(mismatched_levels)
        throw(ArgumentError("contrasts levels not found in data or vice-versa: " *
                            "$mismatched_levels." *
                            "\n  Data levels: $levels." *
                            "\n  Contrast levels: $c_levels"))
    end

    n = length(c_levels)
    if n == 0
        throw(ArgumentError("empty set of levels found (need at least two to compute " *
                            "contrasts)."))
    elseif n == 1
        throw(ArgumentError("only one level found: $(c_levels[1]) (need at least two to " *
                            "compute contrasts)."))
    end

    # find index of base level. use contrasts.base, then default (1).
    baseind = contrasts.base === nothing ?
              1 :
              findfirst(isequal(contrasts.base), c_levels)
    if baseind === nothing
        throw(ArgumentError("base level $(contrasts.base) not found in levels " *
                            "$c_levels."))
    end

    tnames = termnames(contrasts, c_levels, baseind)

    mat = contrasts_matrix(contrasts, baseind, n)

    ContrastsMatrix(mat, tnames, c_levels, contrasts)
end

ContrastsMatrix(c::Type{<:AbstractContrasts}, levels::AbstractVector) =
    throw(ArgumentError("contrast types must be instantiated (use $c() instead of $c)"))

# given an existing ContrastsMatrix, check that all passed levels are present
# in the contrasts. Note that this behavior is different from the
# ContrastsMatrix constructor, which requires that the levels be exactly the same.
# This method exists to support things like `predict` that can operate on new data
# which may contain only a subset of the original data's levels. Checking here
# (instead of in `modelmat_cols`) allows an informative error message.
function ContrastsMatrix(c::ContrastsMatrix, levels::AbstractVector)
    if !isempty(setdiff(levels, c.levels))
         throw(ArgumentError("there are levels in data that are not in ContrastsMatrix: " *
                             "$(setdiff(levels, c.levels))" *
                             "\n  Data levels: $(levels)" *
                             "\n  Contrast levels: $(c.levels)"))
    end
    return c
end

function termnames(C::AbstractContrasts, levels::AbstractVector, baseind::Integer)
    not_base = [1:(baseind-1); (baseind+1):length(levels)]
    levels[not_base]
end

# Making a contrast type T only requires that there be a method for
# contrasts_matrix(T,  baseind, n) and optionally termnames(T, levels, baseind)
# The rest is boilerplate.
for contrastType in [:DummyCoding, :EffectsCoding, :HelmertCoding]
    @eval begin
        mutable struct $contrastType <: AbstractContrasts
            base::Any
            levels::Union{Vector,Nothing}
        end
        ## constructor with optional keyword arguments, defaulting to nothing
        $contrastType(; base=nothing, levels=nothing) = $contrastType(base, levels)
    end
end

mutable struct FullDummyCoding <: AbstractContrasts
# Dummy contrasts have no base level (since all levels produce a column)
end

ContrastsMatrix(C::FullDummyCoding, levels::AbstractVector) =
    ContrastsMatrix(Matrix(1.0I, length(levels), length(levels)), levels, levels, C)

Base.convert(::Type{ContrastsMatrix{FullDummyCoding}}, C::ContrastsMatrix) =
    ContrastsMatrix(FullDummyCoding(), C.levels)


contrasts_matrix(C::DummyCoding, baseind, n) =
    Matrix(1.0I, n, n)[:, [1:(baseind-1); (baseind+1):n]]



function contrasts_matrix(C::EffectsCoding, baseind, n)
    not_base = [1:(baseind-1); (baseind+1):n]
    mat = Matrix(1.0I, n, n)[:, not_base]
    mat[baseind, :] .= -1
    return mat
end



function contrasts_matrix(C::HelmertCoding, baseind, n)
    mat = zeros(n, n-1)
    for i in 1:n-1
        mat[1:i, i] .= -1
        mat[i+1, i] = i
    end

    # re-shuffle the rows such that base is the all -1.0 row (currently first)
    mat = mat[[baseind; 1:(baseind-1); (baseind+1):end], :]
    return mat
end


mutable struct ContrastsCoding <: AbstractContrasts
    mat::Matrix
    base::Any
    levels::Union{Vector,Nothing}

    function ContrastsCoding(mat, base, levels)
        if levels !== nothing
            check_contrasts_size(mat, length(levels))
        end
        new(mat, base, levels)
    end
end

check_contrasts_size(mat::Matrix, n_lev) =
    size(mat) == (n_lev, n_lev-1) ||
    throw(ArgumentError("contrasts matrix wrong size for $n_lev levels. " *
                        "Expected $((n_lev, n_lev-1)), got $(size(mat))"))

## constructor with optional keyword arguments, defaulting to nothing
ContrastsCoding(mat::Matrix; base=nothing, levels=nothing) =
    ContrastsCoding(mat, base, levels)

function contrasts_matrix(C::ContrastsCoding, baseind, n)
    check_contrasts_size(C.mat, n)
    C.mat
end
