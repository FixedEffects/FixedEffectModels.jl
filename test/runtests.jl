using FixedEffectModels

tests = ["areg", "regife"]


println("Running tests:")

for test in tests
    try
        include(test)
        println("\t\033[1m\033[32mPASSED\033[0m: $(test)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(test)")
        if fatalerrors
            rethrow(e)
        elseif !quiet
            showerror(STDOUT, e, backtrace())
            println()
        end
    end
end

