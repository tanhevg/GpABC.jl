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

      prob = ODEProblem(ODE_3GeneReg, x0 ,Tspan, params)
      Obs = solve(prob, solver, saveat=saveat)

      return Obs
    end

    reference_data = GeneReg(true_params, Tspan, x0, solver, saveat)
    simulator_function(var_params) = GeneReg(vcat(var_params, true_params[n_var_params+1:end]), Tspan, x0, solver, saveat)

    sim_abcsmc_input = SimulatedABCSMCInput(n_var_params,
        n_particles,
        threshold_schedule,
        priors,
        distance_metric,
        simulator_function)

    sim_abcsmc_res = ABCSMC(sim_abcsmc_input, reference_data, write_progress = false)
    @test size(sim_abcsmc_res.population, 1) > 0

    function get_training_data(n_design_points,
        priors,
        simulator_function, distance_metric,
        reference_data)

        X = zeros(n_design_points, length(priors))
        y = zeros(n_design_points)
        for i in 1:n_design_points
            dp = [rand(d) for d in priors]
            X[i,:] = dp
            y[i] = distance_metric(simulator_function(dp), reference_data)
        end

        return X, y
    end

    X, y = get_training_data(n_design_points, priors, simulator_function, distance_metric, reference_data)

    gpem = GPModel(training_x=X, training_y=y, kernel=SquaredExponentialArdKernel())
    gp_train(gpem)

    function predict_distance(p::AbstractArray{Float64})
        result = gp_regression(p,gpem)[1]
        return result
    end

    emu_abcsmc_input = EmulatedABCSMCInput(n_var_params,
        n_particles,
        threshold_schedule,
        priors,
        predict_distance,
        batch_size,
        max_iter)

    emu_abcsmc_res = ABCSMC(emu_abcsmc_input, reference_data, write_progress=false)
    @test size(emu_abcsmc_res.population, 1) > 0

end
