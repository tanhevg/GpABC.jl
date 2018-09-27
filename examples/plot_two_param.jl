using Distributions, DifferentialEquations

idx1 = 1
idx2 = 2

true_params =  [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
priors = [Uniform(0., 5.), Uniform(0., 5.), Uniform(10., 20.),
            Uniform(0., 2.), Uniform(0., 2.), Uniform(0., 2.),
            Uniform(75., 125.),
            Uniform(0., 2.), Uniform(0., 2.), Uniform(0., 2.)]
param_indices = [idx1, idx2]
Tspan = (0.0, 10.0)
x0 = [3.0, 2.0, 1.0]
solver = RK4()
saveat = 0.1

function ODE_3GeneReg(dx, x, pars, t)
    dx[1] = pars[1]/(1+pars[7]*x[3]) - pars[4]*x[1]
    dx[2] = pars[2]*pars[8]*x[1]./(1+pars[8]*x[1]) - pars[5]*x[2]
    dx[3] = pars[3]*pars[9]*x[1]*pars[10]*x[2]./(1+pars[9]*x[1])./(1+pars[10]*x[2]) - pars[6]*x[3]
end

function simulate_particle(param1::T, param2::T) where {T<:Real}
    params = copy(true_params)
    params[idx1] = param1
    params[idx2] = param2

    prob = ODEProblem(ODE_3GeneReg, x0, Tspan, params)
    Obs = solve(prob, solver, saveat=saveat)

    return Obs
end

function simulate_particles(param1::AbstractArray{T, 1}, param2::AbstractArray{T, 1}) where {T<:Real}
    map(x->simulate_particle(x...), Iterators.product(param1, param2))
end

function simulate_prior(count1::Int, count2::Int)
    simulate_particles(rand(priors[idx1], count1), rand(priors[idx2], count2))
end

import GpABC.keep_all_summary_statistic
import Distances.euclidean

function build_distances(model_results, ref_data::AbstractArray{T,2},
    summary_stats_func::Function=keep_all_summary_statistic,
    distance_func::Function=Distances.euclidean) where {T<:Real}
    ref_stats = summary_stats_func(ref_data)
    return map(x->distance_func(x, ref_stats), summary_stats_func.(model_results))
end

function simulate_prior_distances(count1::Int, count2::Int,
    ref_data::AbstractArray{T,2},
    summary_stats_func::Function=keep_all_summary_statistic,
    distance_func::Function=Distances.euclidean) where {T<:Real}
    model_results=simulate_prior(count1, count2)
    return build_distances(model_results, ref_data, summary_stats_func, distance_func)
end

reference_data = simulate_model(true_params[idx1], true_params[idx2])
x = simulate_prior_distances(5, 6, reference_data)
