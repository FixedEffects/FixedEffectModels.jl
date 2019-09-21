using FixedEffectModels

tests = ["reg.jl", 
		 "RegressionResult.jl", 
		 "partial_out.jl"
		 ]

println("Running tests:")

global methods_vec = [:lsmr, :lsmr_parallel, :lsmr_threads]
if Base.USE_GPL_LIBS
	push!(methods_vec,  :cholesky, :qr)
end
try 
    using CuArrays
    push!(methods_vec, :lsmr_gpu)
catch e
    @info "CuArrays not found, skipping test of :lsmr_gpu"
end


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
@test FixedEffectModels.pinvertible(Symmetric([1.0 1.0; 1.0 1.0])) ≈ [1.0 1.0; 1.0 1.0]