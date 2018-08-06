using GpABC, DifferentialEquations, Distances, Distributions
using PyCall, PyPlot; @pyimport pandas as pd
using PyCall, PyPlot; @pyimport seaborn as sns

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
# pyplot()
# plot_x = emu_out.population[2][:, 2]
# plot_y = emu_out.population[2][:, 3]
# plot_z = zeros(length(plot_x), length(plot_y))
# using PyCall
# @pyimport matplotlib.pyplot as plt
# plt.figure()
# plt.tricontour(plot_x, plot_y, plot_z, nlevel=5)
# plt.show()
#
# # plot(emu_out, population_colors=["blue", "green", "black"])
# # plot(emu_out)
# # plot(sim_out, population_colors=["blue", "green", "black"])
# # plot(sim_out)


emu_plot_data = pd.DataFrame(data= Dict( :param_1=>emu_out.population[3][:,1],:param_2=>emu_out.population[3][:,2],:param_3=>emu_out.population[3][:,3]))
sim_plot_data = pd.DataFrame(data= Dict( :param_1=>sim_out.population[3][:,1],:param_2=>sim_out.population[3][:,2],:param_3=>sim_out.population[3][:,3]))
PyPlot.ion()
fig, ax = PyPlot.subplots(figsize=(15,15),ncols=3, nrows=3)
     left   =  0.125  # the left side of the subplots of the figure
     right  =  0.9    # the right side of the subplots of the figure
     bottom =  0.1    # the bottom of the subplots of the figure
     top    =  0.9    # the top of the subplots of the figure
     wspace =  .5     # the amount of width reserved for blank space between subplots
     hspace =  .5    # the amount of height reserved for white space between subplots
     PyPlot.subplots_adjust(
         left    =  left,
         bottom  =  bottom,
         right   =  right,
         top     =  top,
         wspace  =  wspace,
         hspace  =  hspace
     )
     PyPlot.ioff()
     cmap = sns.cubehelix_palette(as_cmap=true, dark=0, light=1, reverse=true)
     # marginals
     sns.distplot(sim_plot_data[:param_1],bins=15, kde= true, hist=true,rug= false ,color = "cornflowerblue",ax=ax[1,1])
     sns.distplot(sim_plot_data[:param_2],bins=15, kde= true, hist=true,rug= false ,   color = "cornflowerblue", ax=ax[2,2])
     sns.distplot(sim_plot_data[:param_3],bins=15, kde= true, hist=true,rug= false ,  color = "cornflowerblue", ax=ax[3,3])
     sns.distplot(emu_plot_data[:param_1],bins=15, kde= true, hist=true,rug= false ,   color = "salmon", label = "lisi",axlabel = "Parameter 1",ax=ax[1,1])
    sns.distplot(emu_plot_data[:param_2],bins=15, kde= true, hist=true,rug= false ,  color = "salmon", label = "lisi",axlabel = "Parameter 2",ax=ax[2,2])
    sns.distplot(emu_plot_data[:param_3],bins=15, kde= true, hist=true,rug= false ,   color = "salmon", label = "lisi",axlabel = "Parameter 3",ax=ax[3,3])
    #joint
     sns.jointplot(x="param_1", y="param_2", data=emu_plot_data, kind="kde",color="lightpink", ax=ax[2,1])

     sns.jointplot(x="param_1", y="param_3", data=emu_plot_data, kind="kde", color="lightpink",ax=ax[3,1])
     sns.jointplot(x="param_2", y="param_1", data=emu_plot_data, kind="kde", color="lightpink",ax=ax[3,2])
     #HERE scatter
     sns.jointplot(y="param_1", x="param_2", data=sim_plot_data, kind="kde", color="cornflowerblue", ax=ax[1,2])
     sns.jointplot(y="param_1", x="param_3", data=sim_plot_data, kind="kde", color="cornflowerblue",ax=ax[1,3])
     sns.jointplot(y="param_2", x="param_1", data=sim_plot_data, kind="kde", color="cornflowerblue",ax=ax[2,3])
     ax[1,2][:set_xlabel]("Parameter 2")
     ax[1,2][:set_ylabel]("Parameter 1")
     #
     ax[1,3][:set_xlabel]("Parameter 3")
     ax[1,3][:set_ylabel]("Parameter 1")
     #
     ax[2,3][:set_xlabel]("Parameter 3")
     ax[2,3][:set_ylabel]("Parameter 2")
     #
     ax[2,1][:set_xlabel]("Parameter 1")
     ax[2,1][:set_ylabel]("Parameter 2")
     #
     ax[3,1][:set_xlabel]("Parameter 1")
     ax[3,1][:set_ylabel]("Parameter 3")
     #
     ax[3,2][:set_xlabel]("Parameter 2")
     ax[3,2][:set_ylabel]("Parameter 3")
PyPlot.show(fig)
