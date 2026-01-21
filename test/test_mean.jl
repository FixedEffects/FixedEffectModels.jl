using FixedEffectModels, GLM
using DataFrames
using Test

function test_common_lm(a,b)
    @test isapprox(r2(a),r2(b))
    @test isapprox(adjr2(a),adjr2(b))
    #@test isapprox(coef(a)[1],coef(b)[1])
end 
treatment_ = ["A","A","A","B","B","B","B"]
     year_ = [0,1,2,       0,1,2,3]

tf=DataFrame(treatment = treatment_, year = year_,
             obs = rand(length(treatment_)))

@testset "R2 comparisons" begin

@testset "default mean compare noFE to GLM" begin
    m = @formula obs~year+treatment
    ols1=fit(LinearModel,m,tf)
    nofesol=reg(tf,m,save=true)
    test_common_lm(ols1,nofesol)
end
@testset "specific mean compare noFE to GLM" begin
    m = @formula obs~1+year+treatment
    ols1=fit(LinearModel,m,tf)
    nofesol=reg(tf,m,save=true)
    test_common_lm(ols1,nofesol)
end
@testset "no mean compare noFE to GLM" begin
    m = @formula obs~0+year+treatment
    ols1=fit(LinearModel,m,tf)
    nofesol=reg(tf,m,save=true)
    test_common_lm(ols1,nofesol)
end
@testset "default mean compare fe(treatment) to GLM" begin
    m   = @formula obs~year+treatment
    mfe = @formula obs~year+fe(treatment)
    
    ols1=fit(LinearModel,m,tf)
    fesol=reg(tf,mfe,save=true)
    
    test_common_lm(ols1,fesol)
end
@testset "specific mean compare fe(treatment) to GLM" begin
    m   = @formula obs~1+year+treatment
    mfe = @formula obs~1+year+fe(treatment)
    
    ols1=fit(LinearModel,m,tf)
    fesol=reg(tf,mfe,save=true)
    
    test_common_lm(ols1,fesol)
end
@testset "no mean compare fe(treatment) to GLM" begin
    m   = @formula obs~0+year+treatment
    mfe = @formula obs~0+year+fe(treatment)
    
    ols1=fit(LinearModel,m,tf)
    fesol=reg(tf,mfe,save=true)
    
    test_common_lm(ols1,fesol)
    
    #=

    nofesol=reg(testframe,@formula(obs~0+year+treatment),save=true)
    @test isapprox(r2(nofesol),r2(ols1))
    @test isapprox(adjr2(nofesol),adjr2(ols1))
    @test isapprox(coef(nofesol),coef(ols1))
    @test isapprox(residuals(nofesol,testframe),residuals(ols1))

    @test isapprox(nofesol.tss, fesol.tss) #this fails
    @test isapprox(fesol.tss, nulldeviance(ols1))

    @test isapprox(nofesol.rss, fesol.rss)
    @test isapprox(nofesol.rss, deviance(ols1))

    nofesolgpu=reg(testframe,@formula(obs~0+year+treatment),method=:gpu, double_precision=true,save=true)
    @test isapprox(r2(nofesolgpu),r2(ols1))
    @test isapprox(adjr2(nofesolgpu),adjr2(ols1))

    fesolgpu=reg(testframe,@formula(obs~0+year+fe(treatment)),method=:gpu, double_precision=true,save=true)
    @test isapprox(r2(fesolgpu),r2(ols1))
    @test isapprox(adjr2(fesolgpu),adjr2(ols1))

    @test isapprox(r2(fesolgpu),r2(fesol))
    @test isapprox(adjr2(fesolgpu),adjr2(fesol))
    @test isapprox(coef(fesolgpu),coef(fesol))
    
    @test isapprox(r2(nofesolgpu),r2(nofesol))
    @test isapprox(adjr2(nofesolgpu),adjr2(nofesol))
    @test isapprox(coef(fesolgpu),coef(fesol))
    
    =#
end

end
