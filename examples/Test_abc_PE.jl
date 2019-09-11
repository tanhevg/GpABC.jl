using GpABC, DifferentialEquations, Distances, Plots, Distributions
#
# ABC settings
#
true_params =  [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
priors = [Uniform(0.2, 5.), Uniform(0.2, 5.), Uniform(10., 20.),
            Uniform(0.2, 2.), Uniform(0.2, 2.), Uniform(0.2, 2.),
            Uniform(75., 125.),
            Uniform(0.2, 2.), Uniform(0.2, 2.), Uniform(0.2, 2.)]
param_indices = [1,3,9]
#param_indices = [2,5,10]
priors = priors[param_indices]

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

  function ODE_3GeneReg(dx, x, par, t)
    dx[1] = par[1]/(1+par[7]*x[3]) - par[4]*x[1]
    dx[2] = par[2]*par[8]*x[1]/(1+par[8]*x[1]) - par[5]*x[2]
    dx[3] = par[3]*par[9]*x[1]*par[10]*x[2]./(1+par[9]*x[1])./(1+par[10]*x[2]) - par[6]*x[3]
  end

  prob = ODEProblem(ODE_3GeneReg, x0 ,Tspan, params)
  Obs = solve(prob, solver, saveat=saveat)

  return Array{Float64, 2}(Obs)
end

function simulator_function(var_params)
    params = copy(true_params)
    params[param_indices] .= var_params
    GeneReg(params, Tspan, x0, solver, saveat)
end

reference_data = simulator_function(true_params[param_indices])
plot(reference_data', xlabel="t", ylabel="C(t)", linewidth=2, labels=["u1(t)", "u2(t)", "u3(t)"])

# Rejection ABC
# Simulation
#
n_particles = 2000
threshold = 1.0
sim_result = SimulatedABCRejection(reference_data, simulator_function, priors, threshold, n_particles;
    write_progress=true)
plot(sim_result)

# Emulation
n_design_points = 500
emu_result = EmulatedABCRejection(reference_data, simulator_function, priors, threshold, n_particles, n_design_points;
    summary_statistic="keep_all", distance_function = Distances.euclidean, write_progress=true)
plot(emu_result)

# ABC-SMC
# Simulation
#
#threshold_schedule = [3.0, 2.0, 1.0, 0.5, 0.2];
threshold_schedule = [1.0, 0.8, 0.6, 0.4, 0.2];
population_colors=["#FF2F4E", "#D0001F", "#A20018", "#990017", "#800013"]

@time sim_abcsmc_res = SimulatedABCSMC(reference_data,
    simulator_function,
    priors,
    threshold_schedule,
    n_particles;
    write_progress = true)

plot(sim_abcsmc_res, population_colors=population_colors)

@time emu_abcsmc_res = EmulatedABCSMC(reference_data,
    simulator_function,
    priors,
    threshold_schedule,
    n_particles,
    n_design_points;
    batch_size=500,
    write_progress=true,
    emulator_retraining = PreviousPopulationThresholdRetraining(n_design_points, 100, 10),
    emulated_particle_selection = MeanVarEmulatedParticleSelection()
    )
plot(emu_abcsmc_res, population_colors=population_colors)
