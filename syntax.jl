



using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])
# Solution 1
@reg df Sales ~ NDI fe = StatePooled + YearPooled weight = Pop vcov = cluster(StatePooled) maxiter(2)
@reg(df, Sales ~ NDI, fe = StatePooled + YearPooled, weight = Pop, vcov = cluster(StatePooled), maxiter(2))
# Solution 1'
@reg df Sales ~ NDI fe(StatePooled + YearPooled) weight(Pop) vcov(cluster(StatePooled)) maxiter(2)

# Solution 2
reg(df, @formula(Sales ~ NDI, fe = StatePooled + YearPooled, weight = Pop, vcov = cluster(StatePooled), where = (NDI >= 3)))
# Solution 2'
reg(df, @formula(Sales ~ NDI, fe(StatePooled + YearPooled), weight(Pop), vcov(cluster(StatePooled)), where((NDI >= 3)))


