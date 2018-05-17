using Base.Test, GpAbc, DifferentialEquations

# Just a basic test that verifies that all functions compile and return correctly shaped arrays
@testset "Rejection ABC test" begin
    simulated_dimensions = 3
    design_point_count = 100
    test_point_count = 1000
    epsilon = 0.2

    static_params = [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
    x0 = [0.0, 0.0, 0.0]
    tspan = (0.0,10.0)

    function euclidean_distance_ode(x1::AbstractArray{<:AbstractArray, 1}, x2::AbstractArray{<:AbstractArray, 1})
        x1 = hcat(x1...)
        x2 = hcat(x2...)
        sum([vecnorm(x1[i, :] - x2[i, :]) for i=1:size(x1, 1)])
    end

    function ODE_3GeneReg(t, x, dx, params)
        dx[1] = params[1]./(1+params[7]*x[3]) - params[4]*x[1]
        dx[2] = params[2]*params[8]*x[1]./(1+params[8]*x[1]) - params[5]*x[2]
        dx[3] = params[3]*params[9]*x[1]*params[10]*x[2]./(1+params[9]*x[1])./(1+params[10]*x[2]) - params[6]*x[3]
    end;

    function ode_simulation(ode_params)
        ode_params = [ode_params; static_params[size(ode_params,1)+1:end]]
        prob = ODEProblem((t, x, dx)->ODE_3GeneReg(t, x, dx, ode_params),
            x0, tspan)
        sol = solve(prob, RK4(), saveat=1.0)
        return sol.u
    end;

    function sumulate_training_y_ode(training_x, observed_data)
        training_y = Array{Float64}(size(training_x, 1))
        for i=1:length(training_y)
            training_point = ode_simulation(training_x[i, :])
            training_y[i] = euclidean_distance_ode(observed_data, training_point)
        end
        training_y
    end;


    observed_data = ode_simulation(static_params)

    prior_lower_boundary = zeros(simulated_dimensions)
    prior_upper_boundary = static_params[1:simulated_dimensions] * 2

    gp_training_prior=LatinHypercubeSampler(prior_lower_boundary, prior_upper_boundary)
    gp_training_x = rand(gp_training_prior, design_point_count)'

    gp_training_y = sumulate_training_y_ode(gp_training_x, observed_data)

    gpem = GPModel(gp_training_x, gp_training_y)
    gp_train(gpem)

    test_prior = MvUniform(prior_lower_boundary, prior_upper_boundary)

    function my_test_prior()
        # sample from Mv Uniform or Latin hypercube
        # pre-process test data here
        # return an array
        rand(test_prior, test_point_count)'
    end

    function my_summary_statistic(x)
        gp_mean, _ = gp_regression(x, gpem)
        gp_mean
        # do the GP Regression using gpem
        # return the GP mean
    end


    emulated_params_1 = rejection_abc(my_test_prior, my_summary_statistic,epsilon=0.8)
    emulated_params_2 = rejection_abc(my_test_prior, my_summary_statistic,return_count = 50)
    emulated_params_3 = rejection_abc(my_test_prior, my_summary_statistic, epsilon=0.5, return_count=200, max_iterations=1000)

    @test size(emulated_params_1, 1) > 0
    @test size(emulated_params_1, 2) == simulated_dimensions
    @test size(emulated_params_2, 1) > 0
    @test size(emulated_params_2, 2) == simulated_dimensions
    @test size(emulated_params_3, 1) > 0
    @test size(emulated_params_3, 2) == simulated_dimensions


end
