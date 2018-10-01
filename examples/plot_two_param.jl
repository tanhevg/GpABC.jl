using Distributions, DifferentialEquations, GpABC, Distances, PyPlot

idx1 = 2
idx2 = 3
n_design_point = 100
n_plot = 100
contour_colors = ["white", "#FFE9EC", "#FFBBC5", "#FF8B9C", "#FF5D75", "#FF2F4E", "#D0001F", "#A20018", "#990017", "#800013"]

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

function simulate_particles_product(param1::AbstractArray{T, 1}, param2::AbstractArray{T, 1}) where {T<:Real}
    map(x->simulate_particle(x...), Iterators.product(param1, param2))
end

function emulate_particles_product(param1::AbstractArray{T, 1}, param2::AbstractArray{T, 1}, gpem::GPModel) where {T<:Real}
    test_x = vcat(map(x->[x...]',
        reshape(collect(Iterators.product(param1, param2)),
            (length(param1) * length(param2),) ))...)
    ret = gp_regression(test_x, gpem)
    return reshape(ret[1], (length(param1), length(param2)))
end

function simulate_particles(params::AbstractArray{T, 2}) where {T<:Real}
    map(i->simulate_particle(params[i, :]...), range(1, size(params, 1)))
end

# function simulate_prior(count1::Int, count2::Int)
#     simulate_particles(rand(priors[idx1], count1), rand(priors[idx2], count2))
# end



function build_distances(model_results, ref_data::AbstractArray{T,2},
    summary_stats_func::Function=GpABC.keep_all_summary_statistic,
    distance_func::Function=Distances.euclidean) where {T<:Real}
    ref_stats = summary_stats_func(ref_data)
    return map(x->distance_func(x, ref_stats), summary_stats_func.(model_results))
end

# function simulate_prior_distances(count1::Int, count2::Int,
#     ref_data::AbstractArray{T,2},
#     summary_stats_func::Function=keep_all_summary_statistic,
#     distance_func::Function=Distances.euclidean) where {T<:Real}
#     model_results=simulate_prior(count1, count2)
#     return build_distances(model_results, ref_data, summary_stats_func, distance_func)
# end

reference_data = simulate_particle(true_params[idx1], true_params[idx2])
contour_x = linspace(params(priors[idx1])..., n_plot)
contour_y = linspace(params(priors[idx2])..., n_plot)
sim_result = build_distances(
    simulate_particles_product(contour_x, contour_y),
    reference_data)

training_x = hcat(rand(priors[idx1], n_design_point), rand(priors[idx2], n_design_point))
training_y = build_distances(simulate_particles(training_x), reference_data)
gpm = GPModel(training_x=training_x, training_y=training_y, kernel=SquaredExponentialArdKernel())
gp_train(gpm)
em_result = emulate_particles_product(contour_x, contour_y, gpm)

subplot(121)
subplots_adjust(wspace  =  0.5)
contourf(contour_x, contour_y, sim_result, 8, colors=contour_colors)
title("L2 norm - Simulation")
xlabel("Parameter $(idx1)")
ylabel("Parameter $(idx2)")
subplot(122)
contourf(contour_x, contour_y, em_result, 8, colors=contour_colors)
scatter(training_x[:, 1], training_x[:,2], marker="x", label="training points")
legend()
xlim(params(priors[idx1])...)
ylim(params(priors[idx2])...)
title("L2 norm - Emulation")
xlabel("Parameter $(idx1)")
ylabel("Parameter $(idx2)")
# x = simulate_prior_distances(5, 6, reference_data)
