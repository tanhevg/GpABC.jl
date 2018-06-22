using Base.Test, GpABC

struct UnitTestKernel <: AbstractGPKernel end;

@testset "Abstract Kernel Tests" begin
    @test :covariance_gradient_not_implemented == covariance_grad(UnitTestKernel(), zeros(0), zeros(0,0), zeros(0,0))
end
