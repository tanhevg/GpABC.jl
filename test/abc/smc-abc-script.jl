using GpABC, DifferentialEquations, Distances, Distributions

srand(2)

#
# ABC settings
#
n_var_params = 3
n_particles = 1000
threshold_schedule = [3.0, 2.0, 1.0]
priors = [Uniform(0., 5.), Uniform(0., 5.), Uniform(0., 30.)]
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

println("SIMULATION")
sim_out = SimulatedABCSMC(reference_data, n_particles, threshold_schedule,
    priors, "keep_all", simulator_function)

println("EMULATION")
emu_out = EmulatedABCSMC(n_design_points, reference_data, n_particles, threshold_schedule,
    priors, "keep_all", simulator_function,
    repetitive_training=RepetitiveTraining(rt_iterations=3, rt_extra_training_points=5))

# using Plots
# plot(emu_out, population_colors=["blue", "green", "black"])
# plot(emu_out)
# plot(sim_out, population_colors=["blue", "green", "black"])
# plot(sim_out)
