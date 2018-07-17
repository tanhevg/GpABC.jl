using Base.Test, GpABC, DifferentialEquations

@testset "LNA Test" begin

    params = [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
    Tspan = (0.0, 10.0)
    volume = 1.0

    #stochiometry matrix
    S = [1.0 -1.0 0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 -1.0 0.0 0.0;
        0.0 0.0 0.0 0.0 1.0 -1.0]

    # manual input of rates of 3 three example
    make_f = function(x,params)
        f = [params[1]/(1+params[7]*x[3]),
            params[4]*x[1],
            params[2]*params[8]*x[1]/(1+params[8]*x[1]),
            params[5]*x[2],
            params[3]*params[9]*x[1]*params[10]*x[2]/(1+params[9]*x[1])/(1+params[10]*x[2]),
            params[6]*x[3]]
        return f
    end

    #LNA Mean Var example
    saveat = 0.001
    solver = DifferentialEquations.DP5()
    x0 = [diagm([0.0, 0.0, 0.0]) ; 0.4*ones(3,3)]
    mean_traj, covar_traj, time_points = lna_covariance(params, Tspan, x0, solver, saveat, S, make_f, volume)
    @test size(mean_traj,2) == length(time_points)
    @test length(covar_traj) == length(time_points)
    @test size(mean_traj,1) == size(S,1)

end
