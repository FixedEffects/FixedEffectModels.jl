model = @model Sales ~ Price weights = Pop fe = pYear save = true
result = reg(df, model)