using Base.Test, GpABC, DifferentialEquations, Distances, Distributions

@testset "Rejection ABC Test" begin
    srand(2)
    #
    # ABC settings
    #
    n_var_params = 2
    n_particles = 1000
    priors = [Uniform(0., 5.), Uniform(0., 5.)]
    distance_metric = euclidean
    progress_every = 1000

    #
    # Emulation settings
    #
    n_design_points = 100
    batch_size = 1000
    max_iter = 1000

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

      prob = ODEProblem(ODE_3GeneReg, x0, Tspan, params)
      Obs = solve(prob, solver, saveat=saveat)

      return Obs
    end

    reference_data = GeneReg(true_params, Tspan, x0, solver, saveat)
    simulator_function(var_params) = GeneReg(vcat(var_params, true_params[n_var_params+1:end]), Tspan, x0, solver, saveat)

    #
    # Test using keep all as summary statistic
    #
    sim_rej_input = SimulatedABCRejectionInput(n_var_params,
                            n_particles,
                            0.5,
                            priors,
                            "keep_all",
                            distance_metric,
                            simulator_function,
                            max_iter)

    sim_result = ABCrejection(sim_rej_input, reference_data)
    @test size(sim_result.population, 1) > 0

    #
    # Test using built-in summary statistics
    #
    sim_rej_input = SimulatedABCRejectionInput(n_var_params,
                            n_particles,
                            3.0,
                            priors,
                            ["mean", "variance", "max", "min", "range", "median",
                            "q1", "q3", "iqr"],
                            distance_metric,
                            simulator_function,
                            max_iter)

    sim_result = ABCrejection(sim_rej_input, reference_data)
    @test size(sim_result.population, 1) > 0

    #
    # Test using custom summary statistic
    #
    function sum_stat(data::AbstractArray{Float64,2})
        return std(data, 2)[:]
    end

    sim_rej_input = SimulatedABCRejectionInput(n_var_params,
                            n_particles,
                            3.0,
                            priors,
                            sum_stat,
                            distance_metric,
                            simulator_function,
                            max_iter)

    sim_result = ABCrejection(sim_rej_input, reference_data)
    @test size(sim_result.population, 1) > 0

    gp_train_function = function(prior_sampling_function::Function)
        GpABC.abc_train_emulator(prior_sampling_function,
                n_design_points,
                GpABC.keep_all_summary_statistic(reference_data),
                simulator_function,
                GpABC.build_summary_statistic("keep_all"),
                distance_metric)
    end

    emu_rej_input = EmulatedABCRejectionInput(n_var_params,
          n_particles,
          1.0,
          priors,
          batch_size,
          100,
          EmulatorTrainingInput(
            n_design_points,
            GpABC.keep_all_summary_statistic(reference_data),
                simulator_function,
                "keep_all",
                distance_metric
          ))

    emu_result = ABCrejection(emu_rej_input, reference_data)
    @test size(emu_result.population, 1) > 0

    # Now repeat using user-level functions
    sim_out = SimulatedABCRejection(reference_data, n_particles, 0.5,
        priors, "keep_all", simulator_function)
    @test size(sim_out.population, 1) > 0

    emu_out = EmulatedABCRejection(n_design_points, reference_data, n_particles, 1.0,
        priors, "keep_all", simulator_function)
    @test size(emu_out.population, 1) > 0

    emu_out = EmulatedABCRejection(n_design_points, reference_data, n_particles, 1.0,
        priors, "keep_all", simulator_function,
        emulator_training = DefaultEmulatorTraining(SquaredExponentialIsoKernel()),
        write_progress=false)
    @test size(emu_out.population, 1) > 0
end
