



using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])
@reg df Sales ~ NDI fe(StatePooled + YearPooled) weight(Pop) vcov(robust)
@reg df Sales ~ NDI fe(StatePooled + YearPooled) weight(Pop) vcov(cluster(StatePooled))
@reg df Sales ~ NDI fe(StatePooled + YearPooled) weight(Pop) vcov(cluster(StatePooled)) maxiter(2)
