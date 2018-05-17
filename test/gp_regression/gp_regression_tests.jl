using Base.Test, GpAbc
import GpAbc.covariance, GpAbc.get_hyperparameters_size

struct NoGradKernel <: AbstractGPKernel
    rbf_iso_kernel::GpAbc.SquaredExponentialIsoKernel
end

function NoGradKernel()
    NoGradKernel(GpAbc.SquaredExponentialIsoKernel())
end

function get_hyperparameters_size(ker::NoGradKernel, x::AbstractArray{Float64, 2})
    get_hyperparameters_size(ker.rbf_iso_kernel, x)
end

function covariance(ker::NoGradKernel, log_theta::AbstractArray{Float64, 1},
    x1::AbstractArray{Float64, 2}, x2::AbstractArray{Float64, 2})
    covariance(ker.rbf_iso_kernel, log_theta, x1, x2)
end


@testset "GP Reression Tests" begin
    training_x = readcsv("$(@__DIR__)/training_x1.csv")
    training_y = readcsv("$(@__DIR__)/training_y1.csv")
    test_x = readcsv("$(@__DIR__)/test_x1.csv")

    gpem = GPModel(training_x, training_y, test_x)

    theta_mle = gp_train(gpem)
    @test [58.7668, 2.94946, 5.84869] ≈ theta_mle rtol=1e-5
    @test -183.80873007668373 ≈ gp_loglikelihood(gpem)
    test_grad = gp_loglikelihood_grad(log.(gpem.gp_hyperparameters), gpem)
    @test all(abs.(test_grad - [46540815059614715e-22, 6451822871511581e-22, 20565826801544063e-22]) .< 1e-5)

    regression_mean = readcsv("$(@__DIR__)/test_y1_mean.csv")
    regression_var = readcsv("$(@__DIR__)/test_y1_var.csv")
    (r_mean, r_var) = gp_regression(gpem, batch_size=10)
    @test r_mean ≈ regression_mean
    @test r_var ≈ regression_var

    regression_cov = readcsv("$(@__DIR__)/test_y1_cov.csv")
    (r_mean, r_cov) = gp_regression(gpem, batch_size=10, full_covariance_matrix=true)
    @test r_cov ≈ regression_cov

    gpem = GPModel(training_x, training_y, NoGradKernel())
    theta_mle = gp_train(gpem)
    @test [58.7668, 2.94946, 5.84869] ≈ theta_mle rtol=1e-3

    # Test that 1-d inputs work
    training_x = reshape(training_x, size(training_x, 1))
    training_y = reshape(training_y, size(training_y, 1))
    gpem = GPModel(training_x, training_y)
    theta_mle = gp_train(gpem)

end
