using Base.Test
@testset "GpABC Full Test Suite" begin
include("kernels/abstract_kernel_tests.jl")
include("kernels/scaled_squared_distance_test.jl")
include("kernels/rbf_kernel_tests.jl")
include("kernels/matern_kernel_test.jl")
include("gp_regression/gp_regression_tests.jl")
include("abc/rejection-abc-test.jl")
include("abc/smc-abc-test.jl")
include("abc/model-selection-test.jl")
include("lna-test.jl")
end
