using CSV, CategoricalArrays, DataFrames, Test
using FixedEffectModels
using FixedEffectModels: parse_fe, parse_fixedeffect, _parse_fixedeffect, _multiply
using FixedEffects
import Base: ==

function ==(x::FixedEffect, y::FixedEffect)
    x.refs == y.refs && x.interaction == y.interaction && x.n == y.n
end

@testset "fixed-effect slopes absorb matching RHS terms" begin
    formula_main, formula_fes = parse_fe(@formula(y ~ z + fe(id)*x))
    @test StatsModels.termvars(formula_main) == [:y, :z]
    @test StatsModels.termvars(formula_fes) == [:id, :x]

    formula_main_explicit, _ = parse_fe(@formula(y ~ z + x + fe(id) + fe(id)&x))
    @test formula_main_explicit == formula_main
end

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

    @test parse_fixedeffect(data, ()) == (FixedEffect[], Symbol[], Symbol[])
    
    f = @formula(y ~ 1 + Price)
    ts1 = f.rhs
    ts2 = term(1) + term(:Price)
    @test parse_fixedeffect(data, f) == (FixedEffect[], Symbol[], Symbol[])
    @test parse_fixedeffect(data, ts1) == (FixedEffect[], Symbol[], Symbol[])
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    fparsed = term(:y) ~ InterceptTerm{false}() + term(:Price)
    tsparsed = (InterceptTerm{false}(), term(:Price))

    f = @formula(y ~ 1 + Price + fe(State))
    ts1 = f.rhs
    ts2 = term(1) + term(:Price) + fe(:State)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State)], [:fe_State], [:State])
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State)], [:fe_State], [:State])
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    f = @formula(y ~ Price + fe(State) + fe(Year))
    ts1 = f.rhs
    ts2 = term(:Price) + fe(:State) + fe(:Year)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State), FixedEffect(data.Year)], [:fe_State, :fe_Year], [:State, :Year])
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State), FixedEffect(data.Year)], [:fe_State, :fe_Year], [:State, :Year])
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    f = @formula(y ~ Price + fe(State)&Year)
    ts1 = f.rhs
    ts2 = term(:Price) + fe(:State)&term(:Year)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State, interaction=_multiply(data, [:Year]))], [Symbol("fe_State&Year")], [:State])
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State, interaction=_multiply(data, [:Year]))], [Symbol("fe_State&Year")], [:State])
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)

    f = @formula(y ~ Price + fe(State)*fe(Year))
    ts1 = f.rhs
    ts2 = term(:Price) + fe(:State) + fe(:Year) + fe(:State)&fe(:Year)
    @test parse_fixedeffect(data, f) == ([FixedEffect(data.State), FixedEffect(data.Year), FixedEffect(data.State, data.Year)], [:fe_State, :fe_Year, Symbol("fe_State&fe_Year")], [:State, :Year])
    @test parse_fixedeffect(data, ts1) == ([FixedEffect(data.State), FixedEffect(data.Year), FixedEffect(data.State, data.Year)], [:fe_State, :fe_Year, Symbol("fe_State&fe_Year")], [:State, :Year])
    @test parse_fixedeffect(data, ts2) == parse_fixedeffect(data, ts1)
end

@testset "fixed effect slope restrictions" begin
    data = DataFrame(y = 1:4, id = [1, 1, 2, 2], x = [1.0, 2.0, 3.0, 4.0],
                     z = [2.0, 3.0, 4.0, 5.0], c = categorical(["a", "b", "a", "b"]))

    fes, feids, fekeys = parse_fixedeffect(data, @formula(y ~ fe(id)&x&z))
    @test length(fes) == 1
    @test feids == [Symbol("fe_id&x&z")]
    @test fekeys == [:id]
    @test fes[1].interaction == data.x .* data.z

    err = try
        parse_fixedeffect(data, @formula(y ~ fe(id)&log(x)))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("only support plain numeric columns", sprint(showerror, err))
    @test occursin("create a numeric column first", sprint(showerror, err))

    err = try
        parse_fixedeffect(data, @formula(y ~ fe(id)&x^2))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("only support plain numeric columns", sprint(showerror, err))

    err = try
        parse_fixedeffect(data, @formula(y ~ fe(id)&c))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("only support numeric columns", sprint(showerror, err))
    @test occursin("convert it to numeric", sprint(showerror, err))

    data_missing = DataFrame(y = 1:4, id = [1, 1, 2, 2],
                             x = [1.0, missing, 3.0, 4.0])
    fes_missing, _, _ = parse_fixedeffect(data_missing, @formula(y ~ fe(id)&x))
    @test fes_missing[1].interaction == [1.0, 0.0, 3.0, 4.0]
end
