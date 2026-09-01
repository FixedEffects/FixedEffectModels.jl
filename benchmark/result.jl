using DataFrames, CSV, StatsPlots
df = CSV.read(joinpath(@__DIR__, "benchmark.csv"), DataFrame)
df."fixest (R)" = df."fixest (R)" ./ df."FixedEffectModels.jl (Julia)"
df."lfe (R)" = df."lfe (R)" ./ df."FixedEffectModels.jl (Julia)"
df."reg / reghdfe (Stata)" = df."reg / reghdfe (Stata)" ./ df."FixedEffectModels.jl (Julia)"
df."FixedEffectModels.jl (Julia)" = df."FixedEffectModels.jl (Julia)" ./ df."FixedEffectModels.jl (Julia)"
mdf = stack(df, Not([:Command, :Order]))
mdf = rename(mdf, :variable => :Language)
p = @df mdf plot(
    :Command, :value,
    group = :Language,
    yaxis = :log10,
    xlabel = "Command",
    ylabel = "Time (Ratio to Julia)",
    legend = :top,
    seriestype = :scatter,
    palette = :tol_light,
    dpi = 200,
    right_margin = 8Plots.mm,
    size = (8 * 100 * 2 / 3, 5 * 100 * 2 / 3))
savefig(joinpath(@__DIR__, "fixedeffectmodels_benchmark.png"))
