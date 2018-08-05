using GpABC, DifferentialEquations, Distances, Distributions

n_var_params = 3
n_particles = 1000
priors = [Uniform(0., 5.), Uniform(0., 5.), Distributions.Uniform(0., 30.)]
priors = priors[1:n_var_params]
distance_metric = euclidean
progress_every = 1000

#
# Emulation settings
#
n_design_points = 1000
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

function generate_parameters(
        priors::AbstractArray{D,1},
        batch_size::Int,
        ) where {
        D<:ContinuousUnivariateDistribution
        }
    n_dims = length(priors)
    parameters = zeros(batch_size, n_dims)
    weights = ones(batch_size)

    for j in 1:n_dims
        for i in 1:batch_size
            @inbounds parameters[i,j] = rand(priors[j])
            weights[i] *= pdf(priors[j], parameters[j])
        end
    end

    return parameters, weights
end

srand(15)
emulator = GpABC.abc_train_emulator(size->generate_parameters(priors, size)[1],
        n_design_points,
        GpABC.keep_all_summary_statistic(reference_data),
        simulator_function,
        GpABC.keep_all_summary_statistic,
        distance_metric)
println(min(emulator.gp_training_y...))

println(sum(emulator.gp_training_y .< 0.5))
# test_data = generate_parameters(priors, 10000)[1]
# mean, variance = gp_regression(test_data, emulator)
# sum(mean .< 0.5)
# variance[find(mean .< 0.5)]
# mean[indmax(variance)]
# sample = gp_regression_sample(test_data, emulator)
# println(sum(sample .< 0.5))

emulation_settings = AbcEmulationSettings(n_design_points,
    x -> emulator,
    gp_regression_sample)

input = EmulatedABCRejectionInput(length(priors), n_particles, 0.5,
    priors, emulation_settings, 10000, 1000)

output = ABCrejection(input, reference_data, progress_every=100)

# using Plots
plot(output)

# emu_out = EmulatedABCRejection(n_design_points, reference_data, n_particles, 0.5,
#     priors, "keep_all", simulator_function)
