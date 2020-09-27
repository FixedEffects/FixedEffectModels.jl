function _precompile_()
    Base.precompile(Tuple{typeof(FixedEffectModels.parse_fixedeffect),DataFrame,FormulaTerm})
    Base.precompile(Tuple{typeof(reg),Any,FormulaTerm})
    Base.precompile(Tuple{typeof(partial_out),Any,FormulaTerm})
    Base.precompile(Tuple{typeof(show),FixedEffectModel})
    Base.precompile(Tuple{typeof(FixedEffectModels.basecol),Array{Float64,2}})
end
