using DataFrames, Gadfly
df = readtable("/Users/Matthieu/Dropbox/Github/FixedEffectModels.jl/benchmark/benchmark.csv")
mdf = melt(df[[:Command, :Julia, :R, :Stata]], :Command)
mdf = rename(mdf, :variable, :Language)
p = plot(mdf, x = "Command", y = "value", color = "Language", Guide.ylabel("Time (seconds)"), Guide.xlabel("Model"), Scale.y_log10)