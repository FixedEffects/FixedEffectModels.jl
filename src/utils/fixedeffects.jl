##############################################################################
##
## Drop Singletons
##
##############################################################################

function drop_singletons!(esample, fes::AbstractVector{<:FixedEffect}, _nthreads::Integer = Threads.nthreads())
    isempty(fes) && return 0

    counts = Vector{Vector{Int}}(undef, length(fes))
    # When a group count is one, the xor of active observation indices identifies
    # the only remaining observation in that group.
    active_xor = Vector{Vector{Int}}(undef, length(fes))
    queue_fe = Int[]
    queue_ref = Int[]

    # Count active observations by FE level and seed the singleton queue.
    @inbounds for j in eachindex(fes)
        fe = fes[j]
        counts[j] = zeros(Int, fe.n)
        active_xor[j] = zeros(Int, fe.n)
        counts_j = counts[j]
        nactive = count_groups!(counts_j, active_xor[j], esample, fe.refs)
        nactive == 0 && return 0

        nsingleton_groups = 0
        @inbounds for ref in eachindex(counts_j)
            if counts_j[ref] == 1
                nsingleton_groups += 1
            end
        end
        if nsingleton_groups == nactive
            fill!(esample, false)
            return nactive
        end
        if nsingleton_groups > 0
            @inbounds for ref in eachindex(counts_j)
                if counts_j[ref] == 1
                    push!(queue_fe, j)
                    push!(queue_ref, ref)
                end
            end
        end
    end
    isempty(queue_fe) && return 0

    nsingletons = 0
    head = 1
    # Peel singleton FE levels until every remaining level has degree at least two.
    @inbounds while head <= length(queue_fe)
        j = queue_fe[head]
        ref = queue_ref[head]
        head += 1
        counts[j][ref] == 1 || continue

        obsindex = active_xor[j][ref]
        obsindex != 0 && esample[obsindex] || continue

        esample[obsindex] = false
        nsingletons += 1
        # Removing one observation can create new singleton levels in every FE.
        for k in eachindex(fes)
            ref_k = fes[k].refs[obsindex]
            counts_k = counts[k]
            active_xor_k = active_xor[k]
            counts_k[ref_k] -= 1
            active_xor_k[ref_k] = xor(active_xor_k[ref_k], obsindex)
            if counts_k[ref_k] == 1
                push!(queue_fe, k)
                push!(queue_ref, ref_k)
            end
        end
    end
    return nsingletons
end

function count_groups!(counts, active_xor, esample, refs)
    nactive = 0
    @inbounds for i in eachindex(esample, refs)
        if esample[i]
            nactive += 1
            ref = refs[i]
            counts[ref] += 1
            active_xor[ref] = xor(active_xor[ref], i)
        end
    end
    return nactive
end

function drop_singletons!(esample, fe::FixedEffect, cache)
    return drop_singletons!(esample, FixedEffect[fe])
end


##############################################################################
##
## Number of distinct values (only ever call when fe without missing values)
## 
##############################################################################

function nunique(fe::FixedEffect)
    seen = falses(fe.n)
    if fe.interaction isa UnitWeights
        @inbounds for ref in fe.refs
            seen[ref] = true
        end
    else
        # Continuous-slope fixed effect (e.g. fe(id)&x): a group whose interaction is
        # identically zero contributes no column to the design and absorbs no degree of
        # freedom (the demeaning sets its scale to 0, SolverCPU.scale!). Count only groups
        # with at least one nonzero interaction value, so dof_fes is not overstated.
        @inbounds for i in eachindex(fe.refs, fe.interaction)
            if !iszero(fe.interaction[i])
                seen[fe.refs[i]] = true
            end
        end
    end
    count(seen)
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
