using CSV, DataFrames, Test
using FixedEffectModels
using FixedEffectModels: parse_fixedeffect, _parse_fixedeffect, _multiply
using FixedEffects
import Base: ==

==(x::FixedEffect{R,I}, y::FixedEffect{R,I}) where {R,I} =
    x.refs == y.refs && x.interaction == y.interaction && x.n == y.n

csvfile = CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))
df = DataFrame(csvfile)

# Any table type supporting the Tables.jl interface should work
for data in [df, csvfile]
	@test _parse_fixedeffect(data, term(:Price)) === nothing
    @test _parse_fixedeffect(data, ConstantTerm(1)) === nothing
    @test _parse_fixedeffect(data, fe(:State)) == (FixedEffect(data.State), :fe_State, [:State])
    
	@test _parse_fixedeffect(data, fe(:State)&term(:Year)) ==
        (FixedEffect(data.State, interaction=_multiply(data, [:Year])), Symbol("fe_State&Year"), [:State])
    @test _parse_fixedeffect(data, fe(:State)&fe(:Year)) ==
        (FixedEffect(data.State, data.Year), Symbol("fe_State&fe_Year"), [:State, :Year])

    @test parse_fixedeffect(data, ()) == (FixedEffect[], Symbol[], Symbol[], ())
    
    f = @formula(y ~ 1 + Price)
    ts1 = f.rhs
    ts2 = term(1) + term(:Price)
    @test parse_fixedeffect(data, f) == (FixedEffect[], Symbol[], Symbol[], f)
    @test parse_fixedeffect(data, ts1) == (FixedEffect[], Symbol[], Symbol[], ts1)
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    fparsed = term(:y) ~ InterceptTerm{false}() + term(:Price)
    tsparsed = (InterceptTerm{false}(), term(:Price))

    f = @formula(y ~ 1 + Price + fe(State))
    ts1 = f.rhs
    ts2 = term(1) + term(:Price) + fe(:State)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State)], [:fe_State], [:State], fparsed)
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State)], [:fe_State], [:State], tsparsed)
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    f = @formula(y ~ Price + fe(State) + fe(Year))
    ts1 = f.rhs
    ts2 = term(:Price) + fe(:State) + fe(:Year)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State), FixedEffect(data.Year)], [:fe_State, :fe_Year], [:State, :Year], fparsed)
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State), FixedEffect(data.Year)], [:fe_State, :fe_Year], [:State, :Year], tsparsed)
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    f = @formula(y ~ Price + fe(State)&Year)
    ts1 = f.rhs
    ts2 = term(:Price) + fe(:State)&term(:Year)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State, interaction=_multiply(data, [:Year]))], [Symbol("fe_State&Year")], [:State], term(:y) ~ (term(:Price),))
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State, interaction=_multiply(data, [:Year]))], [Symbol("fe_State&Year")], [:State], (term(:Price),))
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    f = @formula(y ~ Price + fe(State)*fe(Year))
    ts1 = f.rhs
    ts2 = term(:Price) + fe(:State) + fe(:Year) + fe(:State)&fe(:Year)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State), FixedEffect(data.Year), FixedEffect(data.State, data.Year)], [:fe_State, :fe_Year, Symbol("fe_State&fe_Year")], [:State, :Year], fparsed)
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State), FixedEffect(data.Year), FixedEffect(data.State, data.Year)], [:fe_State, :fe_Year, Symbol("fe_State&fe_Year")], [:State, :Year], tsparsed)
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)
end
