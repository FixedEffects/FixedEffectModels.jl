model = @model Sales ~ Price fe = pState
result = reg(df, model, save = :fe)
@test :residuals ∉ names(result.augmentdf)
@test :pState ∈ names(result.augmentdf)

