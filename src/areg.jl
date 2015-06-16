using DataFrames, GLM


function areg(f::Formula, df::AbstractDataFrame, absorb::Formula)
	# get all variables present in f or absorb (except nothing)
	allvars = [DataFrames.allvars(f), DataFrames.allvars(absorb)]
	cols = setdiff(allvars, [:nothing])

	# demean variables 
	dfm = demean(df, DataFrames.allvars(f), absorb)

	# Remove intercept from f 
	terms = DataFrames.Terms(f)
	terms.intercept = false

	# Estimate linear model
	mf = ModelFrame(terms, dfm)
	mm = ModelMatrix(mf)
	y = model_response(mf)
    fit(GLM.LinearModel, mm.m, y)
end








