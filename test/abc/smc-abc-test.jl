using Test, GpABC, OrdinaryDiffEq, Distances, Distributions

@testset "SMC ABC Test" begin
    #
    # ABC settings
    #
    n_particles = 1000
    threshold_schedule = [3.0, 2.0, 1.0]
    priors = [Uniform(0., 5.), Uniform(0., 5.)]
    progress_every = 1000

    #
    # Emulation settings
    #
    n_design_points = 100

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
    # Returns the solution to the toy model as solved by OrdinaryDiffEq
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

      return Array{Float64, 2}(Obs)
    end

    reference_data = GeneReg(true_params, Tspan, x0, solver, saveat)
    simulator_function(var_params) = GeneReg(vcat(var_params, true_params[length(var_params)+1:end]), Tspan, x0, solver, saveat)

    #
    # Test using keep all as summary statistic
    #
    sim_out = SimulatedABCSMC(reference_data, simulator_function, priors, threshold_schedule, n_particles)
    @test size(sim_out.population, 1) > 0
    emu_out = EmulatedABCSMC(reference_data, simulator_function, priors, threshold_schedule, n_particles, n_design_points)
    @test size(emu_out.population, 1) > 0

    #
    # Test using built-in summary statistics
    #
    sum_stats = ["mean", "variance", "max", "min", "range", "median", "q1", "q3", "iqr"]
    sim_out = SimulatedABCSMC(reference_data, simulator_function, priors, threshold_schedule, n_particles;
        summary_statistic=sum_stats)
    @test size(sim_out.population, 1) > 0
    emu_out = EmulatedABCSMC(reference_data, simulator_function, priors, threshold_schedule, n_particles, n_design_points;
        summary_statistic=sum_stats)
    @test size(emu_out.population, 1) > 0

    #
    # Test using custom summary statistic
    #
    function sum_stat(data::AbstractArray{Float64,2})
        return std(data, dims=2)[:]
    end
    sim_out = SimulatedABCSMC(reference_data, simulator_function, priors, 3.0 * threshold_schedule, n_particles;
        summary_statistic=sum_stat, write_progress=false)
    @test size(sim_out.population, 1) > 0
    emu_out = EmulatedABCSMC(reference_data, simulator_function, priors, 3.0 * threshold_schedule, n_particles, n_design_points;
        summary_statistic=sum_stat, write_progress=false)
    @test size(emu_out.population, 1) > 0

    #
    # Custom emulator (re-)training and particle selection
    #
    emu_out = EmulatedABCSMC(reference_data, simulator_function, priors, 3.0 * threshold_schedule, n_particles, n_design_points;
        summary_statistic=sum_stat,
        emulator_training=DefaultEmulatorTraining(SquaredExponentialIsoKernel()),
        emulator_retraining=IncrementalRetraining(10, 100),
        emulated_particle_selection=MeanVarEmulatedParticleSelection(2.0))
    @test size(emu_out.population, 1) > 0

    for selection in [
        MeanEmulatedParticleSelection(),
        MeanVarEmulatedParticleSelection(),
        MeanVarEmulatedParticleSelection(2.0),
        PosteriorSampledEmulatedParticleSelection(),
        PosteriorSampledEmulatedParticleSelection(true)]

        emu_out = EmulatedABCSMC(reference_data, simulator_function, priors, 3.0 * threshold_schedule, n_particles, n_design_points;
            summary_statistic=sum_stat,
            emulator_training=DefaultEmulatorTraining(SquaredExponentialIsoKernel()),
            emulator_retraining=IncrementalRetraining(10, 100),
            emulated_particle_selection=selection)
        @test size(emu_out.population, 1) > 0

    end

end
