using RDatasets, DataFrames, FixedEffectModels, GLM, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


function glm_helper(formula::Formula, df::DataFrame) 
    model_response(ModelFrame(formula, df)) - predict(glm(formula, df, Normal(), IdentityLink()))
end
function glm_helper(formula::Formula, df::DataFrame, wts::Symbol) 
    model_response(ModelFrame(formula, df)) - predict(glm(formula, df, Normal(), IdentityLink(), wts = convert(Array{Float64}, df[wts])))
end

test = (
    convert(Array{Float64}, partial_out(Sales + Price ~ NDI, df)),
    convert(Array{Float64}, partial_out(Sales + Price ~ NDI |> pState, df)),
    convert(Array{Float64}, partial_out(Sales + Price ~ 1 |> pState, df)),
    convert(Array{Float64}, partial_out(Sales + Price ~ 1, df)),
    mean(convert(Array{Float64}, partial_out(Sales + Price ~ NDI, df, add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(Sales + Price ~ NDI |> pState, df, add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(Sales + Price ~ 1 |> pState, df, add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(Sales + Price ~ 1, df, add_mean = true)), 1),
    convert(Array{Float64}, partial_out(Sales + Price ~ NDI, df, weight = :Pop)),
    convert(Array{Float64}, partial_out(Sales + Price ~ NDI |> pState, df, weight = :Pop)),
    convert(Array{Float64}, partial_out(Sales + Price ~ 1 |> pState, df, weight = :Pop)),
    convert(Array{Float64}, partial_out(Sales + Price ~ 1, df, weight = :Pop)),
    )

answer = (
    hcat(glm_helper(Sales ~ NDI, df), glm_helper(Price ~ NDI, df)),
    hcat(glm_helper(Sales ~ NDI + pState, df), glm_helper(Price ~ NDI + pState, df)),
    hcat(glm_helper(Sales ~ pState, df), glm_helper(Price ~ pState, df)),
    hcat(glm_helper(Sales ~ 1, df), glm_helper(Price ~ 1, df)),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(glm_helper(Sales ~ NDI, df, :Pop), glm_helper(Price ~ NDI, df, :Pop)),
    hcat(glm_helper(Sales ~ NDI + pState, df, :Pop), glm_helper(Price ~ NDI + pState, df, :Pop)),
    hcat(glm_helper(Sales ~ pState, df, :Pop), glm_helper(Price ~ pState, df, :Pop)),
    hcat(glm_helper(Sales ~ 1, df, :Pop), glm_helper(Price ~ 1, df, :Pop))
    )

for i in 1:12
    @test_approx_eq_eps test[i] answer[i]	1e-5
end


df[1, :Sales] = NA
df[2, :Price]  = NA
df[5, :Pop]  = NA
df[6, :Pop]  = -1.0
partial_out(Sales + Price ~ 1, df, weight = :Pop)