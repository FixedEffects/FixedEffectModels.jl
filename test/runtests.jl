fatalerrors = length(ARGS) > 0 && ARGS[1] == "-f"
quiet = length(ARGS) > 0 && ARGS[1] == "-q"
anyerrors = false
tests = ["reg.jl", "RegressionResult.jl", "partial_out.jl", "utils.jl", "regife.jl"]


println("Running tests:")
for t in tests
    try
        include(t)
        println("\t\033[1m\033[32mPASSED\033[0m: $t")
    catch
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $t")
    end
end
