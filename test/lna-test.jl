using Base.Test, GpABC, DifferentialEquations, Distributions, Distances

@testset "LNA Test" begin

    params = [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
    Tspan = (0.0, 10.0)
    volume = 1.0
    n_samples = 10
    n_design_points = 200
    priors = [Uniform(0., 5.), Uniform(0., 5.), Uniform(0., 5.)]
    distance_metric = euclidean
    saveat = 0.1

    #stochiometry matrix
    S = [1.0 -1.0 0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 -1.0 0.0 0.0;
        0.0 0.0 0.0 0.0 1.0 -1.0]

    # manual input of rates of 3 three example
    reaction_rate_function = function(x,params)
        f = [params[1]/(1+params[7]*x[3]),
            params[4]*x[1],
            params[2]*params[8]*x[1]/(1+params[8]*x[1]),
            params[5]*x[2],
            params[3]*params[9]*x[1]*params[10]*x[2]/(1+params[9]*x[1])/(1+params[10]*x[2]),
            params[6]*x[3]]
        return f
    end

    #LNA Mean Var example
    x0 = ([0.0, 0.0, 0.0], 0.4*eye(3,3))
    input = LNAInput(params, S, reaction_rate_function, volume)
    lna = compute_LNA(input, x0, Tspan, saveat)
    @test size(lna.traj_means,2) == length(lna.time_points)
    @test length(lna.traj_covars) == length(lna.time_points)
    @test size(lna.traj_means,1) == size(input.S,1)

    lna_trajectories = sample_LNA_trajectories(lna, n_samples)
    @test size(lna_trajectories,1) == size(input.S,1)
    @test size(lna_trajectories,2) == length(lna.time_points)

    X, y = get_training_data(input, n_samples, n_design_points, priors, "keep_all",
        distance_metric, lna_trajectories, x0, Tspan, saveat)

    @test size(X,1) == n_design_points
    @test size(X,2) == length(priors)

end
