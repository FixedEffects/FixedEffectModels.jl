using FixedEffectModels

tests = ["fit.jl", 
		 "predict.jl", 
		 "partial_out.jl"
		 ]

println("Running tests:")


for test in tests
	try
		include(test)
		println("\t\033[1m\033[32mPASSED\033[0m: $(test)")
	 catch e
	 	println("\t\033[1m\033[31mFAILED\033[0m: $(test)")
	 	showerror(stdout, e, backtrace())
	 	rethrow(e)
	 end
end


using Test
@test Vcov.pinvertible(Symmetric([1.0 1.0; 1.0 1.0])) ≈ [1.0 1.0; 1.0 1.0]