using DataFrames, Distances


type FactorStructure
	id::Fe
	time::Fe
	lambda::Matrix{Float64} 
	f::Matrix{Float64}
end


type FactorEstimate
	beta::Vector{Float64} 
	factor::FactorStructure
end


# demean factors (Bai 2009)
function regife(f::Formula, df::AbstractDataFrame, factors::Formula, absorb::Formula, d::Int64)
  # construct subdataframe wo NA
  allvars = [DataFrames.allvars(f), DataFrames.allvars(factors), DataFrames.allvars(absorb)]
  allvars = setdiff(allvars, [:nothing])
  condition = complete_cases(df[allvars])
  df = df[condition, allvars]
  regife_helper(f, demean!(df, DataFrames.allvars(f), absorb), factors, d)
end

function regife(f::Formula, df::AbstractDataFrame, factors::Formula, d::Int64)
  # construct subdataframe wo NA
  allvars = [DataFrames.allvars(f), DataFrames.allvars(factors)]
  allvars = setdiff(allvars, [:nothing])
  condition = complete_cases(df[allvars])
  df = df[condition, allvars]
  regife_helper(f, demean!(df, DataFrames.allvars(f)), factors, d)
end

function regife_helper(f::Formula, df::AbstractDataFrame, factors::Formula, d::Int64)
    
    # Remove intercept from f 
    terms = DataFrames.Terms(f)
    terms.intercept = false

    mf = ModelFrame(terms, df)
    X = ModelMatrix(mf).m
    Y = model_response(mf)
    temp = At_mul_B(X, X)
    M = A_mul_Bt(inv(cholfact!(temp)), X)
    b = M * Y
    res_vector = Array(Float64, length(Y))

    # construct an array of factors
    id = df[factors.rhs.args[2]]
    id = Fe(id)
    time = df[factors.rhs.args[3]]
    time = Fe(time)
    

    res_matrix = fill(0.0, (length(id.size), length(time.size)))
    Lambda = Array(Float64, (length(id.size), d))
    F = Array(Float64, (length(time.size), d) )
    max_iter = 10000
    tolerance = 1e-7 * length(Y)
    iter = 0
   	while iter < max_iter
   	   iter += 1
       oldb = copy(b)
       # Compute predicted(regressor)
       A_mul_B!(res_vector, X, b)
      for i in 1:length(Y)
            @inbounds res_matrix[id.refs[i], time.refs[i]] = Y[i] - res_vector[i]
       end
       svdresult = svdfact!(res_matrix) 
       A_mul_B!(Lambda, sub(svdresult.U, :, 1:d), diagm(sub(svdresult.S, 1:d)))
       A_mul_B!(res_matrix, Lambda, sub(svdresult.Vt, 1:d, :))
      for i in 1:length(Y)
            @inbounds res_vector[i] = Y[i] - res_matrix[id.refs[i], time.refs[i]]
       end
       # regress Y - predicted(factor) over X
       b = M * res_vector
       error = euclidean(b, oldb)
       if error < tolerance
		    F = transpose(sub(svdresult.Vt, 1:d, :))
		    broadcast!(.* , Lambda, Lambda, 1 / sqrt(length(time.size)))
		    broadcast!(.* , F, F, sqrt(length(time.size)))
           break
       end
   end
   FactorEstimate(b, FactorStructure(id, time, Lambda, F))
end



