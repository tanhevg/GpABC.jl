using GpABC, Distributions, Distances, DifferentialEquations
# using RecipesBase


# using PyPlot
function hei_plot(result, params=nothing, colors=nothing)
      if colors === nothing
            colors = ["green", "blue", "red"]
      end
      if params === nothing
            params = [i for i in 1:result.n_params]
      end
      plot_count = length(params)
      fig, axes = subplots(plot_count,plot_count)
      for (i, par1) in enumerate(params)
            for (j, par2) in enumerate(params)
                  plot_index = (i - 1) * length(params) + j
                  ax = axes[plot_index]
                  if j < i
                        ax[:axis]("off")
                  elseif j == i
                        ax[:hist](result.population[end][:, par1], 50)
                        ax[:set_xlabel]("θ$(par1)")
                        ax[:spines]["top"][:set_color]("none")
                        ax[:spines]["right"][:set_color]("none")
                  else
                        for k in 1:length(colors)
                              pop = result.population[length(result.population) - length(colors) + k]
                              ax[:scatter](pop[:, par1], pop[:, par2], c=colors[k], alpha=0.5, s=4)
                              ax[:set_xlabel]("θ$(par1)")
                              ax[:set_ylabel]("θ$(par2)")
                              ax[:grid]("on")
                        end
                        ax[:spines]["top"][:set_color]("none")
                        ax[:spines]["right"][:set_color]("none")
                  end
            end
      end

end

#=
@recipe function abc_output_recipe(abco::ABCOutput, params=nothing)
      colors = (markercolor --> ["green" "blue" "red"])
      legend := false
      if params === nothing
            params = [i for i in 1:abco.n_params]
      end
      # layout := [1, 2]
      layout := length(params) ^ 2
      for (i, par1) in enumerate(params)
            for (j, par2) in enumerate(params)
                  @series begin
                        subplot := (i - 1) * length(params) + j
                        data = [0]
                        if i == j
                              seriestype := :histogram
                              bins := 50
                              data = abco.population[end][:, par1]
                        elseif j < i
                              seriestype := :scatter
                              x = Vector(length(colors))
                              y = Vector(length(colors))
                              for k in 1:length(colors)
                                    pop = abco.population[length(abco.population) - length(colors) + k]
                                    x[k] = pop[:, par1]
                                    y[k] = pop[:, par2]
                              end # for
                              data = (x, y)
                        else
                              foreground_color_subplot := ColorTypes.RGBA(0, 0, 0, 0)
                              grid := false
                        end # if/elseif
                        data
                  end # @series
            end # for j, par2
      end # for i, par1
end # @recipe
=#
#
# ABC settings
#
n_var_params = 3
n_particles = 1000
threshold_schedule = [3.0, 2.0, 1.0]
priors = [Uniform(0., 5.), Uniform(0., 5.), Uniform(0., 30.)]
# priors = [Uniform(0., 5.), Uniform(0., 5.)]
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

#
# Test using keep all as summary statistic
#
sim_abcsmc_input = SimulatedABCSMCInput(n_var_params,
    n_particles,
    threshold_schedule,
    priors,
    "keep_all",
    distance_metric,
    simulator_function)

sim_abcsmc_res = ABCSMC(sim_abcsmc_input, reference_data, write_progress = false)
# hei_plot(sim_abcsmc_res)
using Plots
gr()
# Plots.scatter([rand(10), rand(10), rand(20)], [rand(10), 2 + rand(10), 4 + rand(20)],
#       color=["pink" "green" "blue"], legend=false)
plot(sim_abcsmc_res)


#=
function get_training_data(n_design_points,
    priors,
    simulator_function, distance_metric,
    reference_data)

    n_var_params = length(priors)

    X = zeros(n_design_points, n_var_params)
    y = zeros(n_design_points)

    for j in 1:n_var_params
        X[:,j] = rand(priors[j], n_design_points)
    end

    for i in 1:n_design_points
        y[i] = distance_metric(simulator_function(X[i,:]), reference_data)
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

emu_abcsmc_res = ABCSMC(emu_abcsmc_input, reference_data)
hei_plot(emu_abcsmc_res)
=#
