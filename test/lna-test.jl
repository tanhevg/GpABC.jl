using Test, GpABC, DifferentialEquations, Distributions, Distances, LinearAlgebra

@testset "LNA Test" begin

    params = [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
    Tspan = (0.0, 10.0)
    volume = 100.0
    n_samples = 10
    n_design_points = 200
    priors = [Uniform(0., 5.), Uniform(0., 5.), Uniform(10., 20.)]
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
    x0 = ([3.0, 2.0, 1.0], 0.4*Matrix(1.0I, 3,3))
    input = LNAInput(params, S, reaction_rate_function, volume)
    lna = compute_LNA(input, x0, Tspan, saveat)
    @test size(lna.traj_means,2) == length(lna.time_points)
    @test length(lna.traj_covars) == length(lna.time_points)
    @test size(lna.traj_means,1) == size(input.S,1)

    lna_trajectories = sample_LNA_trajectories(lna, n_samples)
    @test size(lna_trajectories,1) == size(input.S,1)
    @test size(lna_trajectories,2) == length(lna.time_points)

    #ABC-SMC simulation with LNA
    threshold_schedule = [4.0, 3.0, 2.0]
    n_samples = 100
    n_particles = 500
    n_var_params = length(priors)
    reference_data = sample_LNA_trajectories(lna, n_samples)

    function simulator_function(var_params)
        input = LNAInput(vcat(var_params, params[n_var_params+1:end]), S, reaction_rate_function, volume)
        return get_LNA_trajectories(input, n_samples, x0, Tspan, saveat)
    end

    sim_abcsmc_res = SimulatedABCSMC(reference_data,
    simulator_function,
    priors,
    threshold_schedule,
    n_particles,
    )

    @test size(sim_abcsmc_res.population, 1) > 0

    #ABC-SMC emulation with LNA

    emu_abcsmc_res = EmulatedABCSMC(reference_data,
    simulator_function,
    priors,
    threshold_schedule,
    n_particles,
    n_design_points,
    )

    @test size(emu_abcsmc_res.population, 1) > 0

end
