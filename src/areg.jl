# algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf
using DataFrames, GLM


function areg(f::Formula, df::AbstractDataFrame, absorb::Formula)
	allvars = [DataFrames.allvars(f), DataFrames.allvars(absorb)]
	cols = setdiff(allvars, [:nothing])
	ans = deepcopy(df[cols])
	ans = demean!(ans, DataFrames.allvars(f), absorb)
	terms = DataFrames.Terms(f)
	terms.intercept = false
	mf = ModelFrame(terms, ans)
	mm = ModelMatrix(mf)
	y = model_response(mf)
    fit(GLM.LinearModel, mm.m, y)
end








