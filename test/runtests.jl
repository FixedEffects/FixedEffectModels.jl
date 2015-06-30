using FixedEffectModels

tests = ["reg", 
		 "RegressionResult", 
		 "partial_out", 
		 "utils", 
		 "regife"]

println("Running tests:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end