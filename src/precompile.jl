let
	df = DataFrame(x = rand(10), y = rand(10), w = abs(rand(10)), 
		id = PooledDataArray(
			RefArray(vcat(fill(one(Int), 5), fill(2 * one(Int), 5))), [1, 2]))
	reg(y~x |> id, df)
	reg(y~1 |> id&x, df)
end