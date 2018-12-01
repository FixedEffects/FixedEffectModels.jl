using DataFrames, Statistics, GLM, Test

df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv"))
df[:pState] = categorical(df[:State])
df[:pYear] = categorical(df[:Year])


function glm_helper(formula::Formula, df::DataFrame) 
    model_response(ModelFrame(formula, df)) - predict(glm(formula, df, Normal(), IdentityLink()))
end
function glm_helper(formula::Formula, df::DataFrame, wts::Symbol) 
    model_response(ModelFrame(formula, df)) - predict(glm(formula, df, Normal(), IdentityLink(), wts = convert(Array{Float64}, df[wts])))
end

test = (
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ NDI))[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ NDI, fe = pState))[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ 1, fe = pState))[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ 1))[1]),
    mean(convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ NDI, add_mean = true))[1]), dims = 1[1]),
    mean(convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ NDI, fe = pState, add_mean = true))[1]), dims = 1[1]),
    mean(convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ 1, fe = pState, add_mean = true))[1]), dims = 1[1]),
    mean(convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ 1, add_mean = true))[1]), dims = 1[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ NDI, weights = Pop))[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ NDI, fe = pState, weights = Pop))[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ 1, fe = pState, weights = Pop))[1]),
    convert(Array{Float64}, partial_out(df, @model(Sales + Price ~ 1, weights = Pop))[1]),
    )

answer = (
    hcat(glm_helper(@formula(Sales ~ NDI), df), glm_helper(@formula(Price ~ NDI), df)),
    hcat(glm_helper(@formula(Sales ~ NDI + pState), df), glm_helper(@formula(Price ~ NDI + pState), df)),
    hcat(glm_helper(@formula(Sales ~ pState), df), glm_helper(@formula(Price ~ pState), df)),
    hcat(glm_helper(@formula(Sales ~ 1), df), glm_helper(@formula(Price ~ 1), df)),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(mean(df[:Sales]), mean(df[:Price])),
    hcat(glm_helper(@formula(Sales ~ NDI), df, :Pop), glm_helper(@formula(Price ~ NDI), df, :Pop)),
    hcat(glm_helper(@formula(Sales ~ NDI + pState), df, :Pop), glm_helper(@formula(Price ~ NDI + pState), df, :Pop)),
    hcat(glm_helper(@formula(Sales ~ pState), df, :Pop), glm_helper(@formula(Price ~ pState), df, :Pop)),
    hcat(glm_helper(@formula(Sales ~ 1), df, :Pop), glm_helper(@formula(Price ~ 1), df, :Pop))
    )

for i in 1:12
    @test test[i] ≈ answer[i] atol = 1e-5
end

