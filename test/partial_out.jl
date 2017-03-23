using RDatasets, DataFrames, FixedEffectModels, GLM, Base.Test

x = dataset("plm", "Cigar")
x[:pState] = pool(x[:State])
x[:pYear] = pool(x[:Year])


function glm_helper(formula::Formula, x::DataFrame) 
    model_response(ModelFrame(formula, x)) - predict(glm(formula, x, Normal(), IdentityLink()))
end
function glm_helper(formula::Formula, x::DataFrame, wts::Symbol) 
    model_response(ModelFrame(formula, x)) - predict(glm(formula, x, Normal(), IdentityLink(), wts = convert(Array{Float64}, x[wts])))
end

test = (
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ NDI), x)),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ NDI |> pState), x)),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ 1 |> pState), x)),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ 1), x)),
    mean(convert(Array{Float64}, partial_out(@formula(Sales + Price ~ NDI), x, add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(@formula(Sales + Price ~ NDI |> pState), x, add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(@formula(Sales + Price ~ 1 |> pState), x, add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(@formula(Sales + Price ~ 1), x, add_mean = true)), 1),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ NDI), x, weight = :Pop)),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ NDI |> pState), x, weight = :Pop)),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ 1 |> pState), x, weight = :Pop)),
    convert(Array{Float64}, partial_out(@formula(Sales + Price ~ 1), x, weight = :Pop)),
    )

answer = (
    hcat(glm_helper(@formula(Sales ~ NDI), x), glm_helper(@formula(Price ~ NDI), x)),
    hcat(glm_helper(@formula(Sales ~ NDI + pState), x), glm_helper(@formula(Price ~ NDI + pState), x)),
    hcat(glm_helper(@formula(Sales ~ pState), x), glm_helper(@formula(Price ~ pState), x)),
    hcat(glm_helper(@formula(Sales ~ 1), x), glm_helper(@formula(Price ~ 1), x)),
    hcat(mean(x[:Sales]), mean(x[:Price])),
    hcat(mean(x[:Sales]), mean(x[:Price])),
    hcat(mean(x[:Sales]), mean(x[:Price])),
    hcat(mean(x[:Sales]), mean(x[:Price])),
    hcat(glm_helper(@formula(Sales ~ NDI), x, :Pop), glm_helper(@formula(Price ~ NDI), x, :Pop)),
    hcat(glm_helper(@formula(Sales ~ NDI + pState), x, :Pop), glm_helper(@formula(Price ~ NDI + pState), x, :Pop)),
    hcat(glm_helper(@formula(Sales ~ pState), x, :Pop), glm_helper(@formula(Price ~ pState), x, :Pop)),
    hcat(glm_helper(@formula(Sales ~ 1), x, :Pop), glm_helper(@formula(Price ~ 1), x, :Pop))
    )

for i in 1:12
    @test_approx_eq_eps test[i] answer[i]	1e-5
end


x[1, :Sales] = NA
x[2, :Price]  = NA
x[5, :Pop]  = NA
x[6, :Pop]  = -1.0
partial_out(@formula(Sales + Price ~ 1), x, weight = :Pop)