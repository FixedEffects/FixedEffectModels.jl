##############################################################################
##
## FixedEffect
##
## The categoricalarray may have pools that are never referred. Note that the pool does not appear in FixedEffect anyway.
##
##############################################################################

struct FixedEffect{R <: Integer, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original CategoricalVector
    interaction::I          # the continuous interaction
    n::Int                  # Number of potential values (= maximum(refs))
end

function FixedEffect(x::CategoricalVector, interaction::AbstractVector = Ones{Float64}(length(x)))
    FixedEffect(x.refs, interaction, length(x.pool))
end

##############################################################################
##
## Subset
##
##############################################################################

getindex(x::FixedEffect, idx) = FixedEffect(x.refs[idx], x.interaction[idx], x.n)

##############################################################################
##
## Remove singletons
##
##############################################################################

function remove_singletons!(esample, x::FixedEffect)
    cache = zeros(Int, x.n)
    for i in 1:length(esample)
        if esample[i]
            cache[x.refs[i]] += 1
        end
    end
    for i in 1:length(esample)
        if esample[i] && cache[x.refs[i]] <= 1
            esample[i] = false
        end
    end
end
