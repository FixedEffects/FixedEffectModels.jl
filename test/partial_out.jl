using DataFrames, FixedEffectModels, GLM, Base.Test

x = readtable(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv.gz"))
x[:pState] = pool(x[:State])
x[:pYear] = pool(x[:Year])


function glm_helper(formula::Formula, x::DataFrame) 
    model_response(ModelFrame(formula, x)) - predict(glm(formula, x, Normal(), IdentityLink()))
end
function glm_helper(formula::Formula, x::DataFrame, wts::Symbol) 
    model_response(ModelFrame(formula, x)) - predict(glm(formula, x, Normal(), IdentityLink(), wts = convert(Array{Float64}, x[wts])))
end

test = (
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ NDI))),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ NDI), @fe(pState))),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ 1), @fe(pState))),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ 1))),
    mean(convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ NDI), add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ NDI), @fe(pState), add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ 1), @fe(pState), add_mean = true)), 1),
    mean(convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ 1), add_mean = true)), 1),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ NDI), @weight(Pop))),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ NDI), @fe(pState), @weight(Pop))),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ 1), @fe(pState), @weight(Pop))),
    convert(Array{Float64}, partial_out(x, @formula(Sales + Price ~ 1), @weight(Pop))),
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
    @test test[i] ≈ answer[i] atol = 1e-5
end


x[1, :Sales] = NA
x[2, :Price]  = NA
x[5, :Pop]  = NA
x[6, :Pop]  = -1.0
partial_out(x, @formula(Sales + Price ~ 1), @weight(Pop))