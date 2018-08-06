using Base.Test, GpABC, DifferentialEquations, Distances, Distributions

@testset "SMC ABC Test" begin
    #
    # ABC settings
    #
    n_var_params = 2
    n_particles = 1000
    threshold_schedule = [3.0, 2.0, 1.0]
    priors = [Uniform(0., 5.), Uniform(0., 5.)]
    distance_metric = euclidean
    progress_every = 1000
    max_iter_emu = 1e4

    #
    # Emulation settings
    #
    n_design_points = 100
    batch_size = 1000
    max_iter_sim = 1000

    #
    # True parameters
    #
    true_params =  [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]

    #
    # ODE solver settings
    #
    Tspan = (0.0, 10.0)
    x0 = [3.0, 2.0, 1.0]
    solver = RK4()
    saveat = 0.1

    #
    # Returns the solution to the toy model as solved by DifferentialEquations
    #
    GeneReg = function(params::AbstractArray{Float64,1},
        Tspan::Tuple{Float64,Float64}, x0::AbstractArray{Float64,1},
        solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64)

      if size(params,1) != 10
        throw(ArgumentError("GeneReg needs 10 parameters, $(size(params,1)) were provided"))
      end

      function ODE_3GeneReg(dx, x, pars, t)
        dx[1] = pars[1]/(1+pars[7]*x[3]) - pars[4]*x[1]
        dx[2] = pars[2]*pars[8]*x[1]./(1+pars[8]*x[1]) - pars[5]*x[2]
        dx[3] = pars[3]*pars[9]*x[1]*pars[10]*x[2]./(1+pars[9]*x[1])./(1+pars[10]*x[2]) - pars[6]*x[3]
      end

      prob = ODEProblem(ODE_3GeneReg, x0 ,Tspan, params)
      Obs = solve(prob, solver, saveat=saveat)

      return Obs
    end

    reference_data = GeneReg(true_params, Tspan, x0, solver, saveat)
    simulator_function(var_params) = GeneReg(vcat(var_params, true_params[n_var_params+1:end]), Tspan, x0, solver, saveat)

    #
    # Test using keep all as summary statistic
    #
    sim_abcsmc_input = SimulatedABCSMCInput(n_var_params,
        n_particles,
        threshold_schedule,
        priors,
        "keep_all",
        distance_metric,
        simulator_function,
        max_iter)

    sim_abcsmc_res = ABCSMC(sim_abcsmc_input, reference_data)
    @test size(sim_abcsmc_res.population, 1) > 0

    #
    # Test using built-in summary statistics
    #
    sim_abcsmc_input = SimulatedABCSMCInput(n_var_params,
        n_particles,
        3.0 * threshold_schedule,
        priors,
        ["mean", "variance", "max", "min", "range", "median",
        "q1", "q3", "iqr"],
        distance_metric,
        simulator_function,
        max_iter)

    sim_abcsmc_res = ABCSMC(sim_abcsmc_input, reference_data)
    @test size(sim_abcsmc_res.population, 1) > 0

    #
    # Test using custom summary statistic
    #
    function sum_stat(data::AbstractArray{Float64,2})
        return std(data, 2)[:]
    end
    sim_abcsmc_input = SimulatedABCSMCInput(n_var_params,
        n_particles,
        3.0 * threshold_schedule,
        priors,
        sum_stat,
        distance_metric,
        simulator_function,
        max_iter)

    sim_abcsmc_res = ABCSMC(sim_abcsmc_input, reference_data)
    @test size(sim_abcsmc_res.population, 1) > 0

    gp_train_function = function(prior_sampling_function::Function)
        GpABC.abc_train_emulator(prior_sampling_function,
                n_design_points,
                GpABC.keep_all_summary_statistic(reference_data),
                simulator_function,
                GpABC.build_summary_statistic("keep_all"),
                distance_metric)
    end

    emu_abcsmc_input = EmulatedABCSMCInput(n_var_params,
        n_particles,
        threshold_schedule,
        priors,
        AbcEmulationSettings(n_design_points, gp_train_function, gp_regression_sample),
        batch_size,
        max_iter_emu)

    emu_abcsmc_res = ABCSMC(emu_abcsmc_input, reference_data, write_progress=false)
    @test size(emu_abcsmc_res.population, 1) > 0

    # Now repeat using user-level functions
    sim_out = SimulatedABCSMC(reference_data, n_particles, threshold_schedule,
        priors, "keep_all", simulator_function, write_progress=false)
    @test size(sim_out.population, 1) > 0

    emu_out = EmulatedABCSMC(n_design_points, reference_data, n_particles, threshold_schedule,
        priors, "keep_all", simulator_function, write_progress=false)
    @test size(emu_out.population, 1) > 0

    emu_out = EmulatedABCSMC(n_design_points, reference_data, n_particles, threshold_schedule,
        priors, "keep_all", simulator_function,
        repetitive_training = RepetitiveTraining(rt_iterations=1, rt_extra_training_points=1),
        write_progress=false)
    @test size(emu_out.population, 1) > 0

end
