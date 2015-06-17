using FixedEffectModels

tests = ["areg", "regife"]

println("Running tests:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end
