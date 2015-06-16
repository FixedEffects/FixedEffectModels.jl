# algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf
using DataFrames, GLM


function areg(f::Formula, df::AbstractDataFrame, absorb::Formula)
	# do a deep copy of all variables present in f or absorb (except nothing)
	allvars = [DataFrames.allvars(f), DataFrames.allvars(absorb)]
	cols = setdiff(allvars, [:nothing])
	df = deepcopy(df[cols])

	# demean variables in f in place
	df = demean!(df, DataFrames.allvars(f), absorb)

	# Remove intercept from f and estimate linear model
	terms = DataFrames.Terms(f)
	terms.intercept = false
	mf = ModelFrame(terms, ans)
	mm = ModelMatrix(mf)
	y = model_response(mf)
    fit(GLM.LinearModel, mm.m, y)
end








