using Base.Test, GpAbc, DifferentialEquations

# Just a basic test that verifies that all functions compile and return correctly shaped arrays
@testset "ABC smc test" begin
#testing smc vs. rejection abc
#result: two plots: smc , rejection abc
#plot also steps in between if you want via: one_k_all_seq_abc_run_accepted


simulated_dimensions = 3
design_point_count = 200
test_point_count = 2_000
#epsilons = sort(collect(0.5:2:10),rev=true)
epsilons = [5, 3.0, 0.8]
return_count = -1
max_iterations= 1000

static_params = [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
x0 = [3.0, 2.0, 1.0]
tspan = (0.0,10.0)

function euclidean_distance_ode(x1::AbstractArray{<:AbstractArray, 1}, x2::AbstractArray{<:AbstractArray, 1})
    x1 = hcat(x1...)
    x2 = hcat(x2...)
    sum([vecnorm(x1[i, :] - x2[i, :]) for i=1:size(x1, 1)])
end

function ODE_3GeneReg(dx, x, params, t)
    dx[1] = params[1]./(1+params[7]*x[3]) - params[4]*x[1]
    dx[2] = params[2]*params[8]*x[1]./(1+params[8]*x[1]) - params[5]*x[2]
    dx[3] = params[3]*params[9]*x[1]*params[10]*x[2]./(1+params[9]*x[1])./(1+params[10]*x[2]) - params[6]*x[3]
end;

function ode_simulation(ode_params)
    ode_params = [ode_params; static_params[size(ode_params,1)+1:end]]
    prob = ODEProblem(ODE_3GeneReg, x0, tspan, ode_params)
    sol = solve(prob, RK4(), saveat=1.0)
    return sol.u
end;

function sumulate_training_y_ode(training_x, observed_data)
    training_y = Array{Float64}(size(training_x, 1))
    for i=1:length(training_y)
        training_point = ode_simulation(training_x[i, :])
        training_y[i] = euclidean_distance_ode(observed_data, training_point)
    end
    training_y
end;

observed_data = ode_simulation(static_params)
observed_summary_statistic = 0
prior_lower_boundary = [0.0,0.0,5.0]
prior_upper_boundary = [5.0,5.0,25.0]

gp_training_prior=LatinHypercubeSampler(prior_lower_boundary, prior_upper_boundary)
gp_training_x = rand(gp_training_prior, design_point_count)'
gp_training_y = sumulate_training_y_ode(gp_training_x, observed_data)

println("Simulation done")
gpem = GPModel(gp_training_x, gp_training_y)
gp_train(gpem)
println("Emulator trained")
init_prior = MvUniform(prior_lower_boundary, prior_upper_boundary)

function my_test_prior()
    rand(init_prior, test_point_count)'
end

function my_summary_statistic(x)
    gp_mean, _ = gp_regression(x, gpem)
    gp_mean
end

new_test_points_x, new_test_points_weights, new_test_points_cdf,one_k_all_seq_abc_run_accepted = smc(init_prior,my_summary_statistic,
 epsilons, test_point_count)

    @test size(new_test_points_x, 1) > 0
    @test size(new_test_points_x, 2) == simulated_dimensions


end
