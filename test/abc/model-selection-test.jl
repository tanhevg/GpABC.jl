using Base.Test, GpABC, DifferentialEquations, Distances, Distributions

@testset "Model selection test" begin

	threshold_schedule = [20, 15, 10, 5, 3, 2.5, 2, 1.7, 1.5]
	max_iter = 1e4
	n_particles = 200

	#
	# Experimental data - from ABCSysBio example at
	# https://github.com/jamesscottbrown/abc-sysbio/blob/master/examples/SBML/Example1/SIR.ipynb
	#
	times = [0.0, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
	values = [[ 20.     ,  10.     ,   0.     ],
	       [  0.12313,  13.16813,   9.42344],
	       [  0.12102,   7.17251,  11.18957],
	       [  0.09898,   2.36466,  10.0365 ],
	       [  0.37887,   0.92019,   6.87117],
	       [  1.00661,   0.61958,   4.44955],
	       [  1.20135,   0.17449,   3.01271],
	       [  1.46433,   0.28039,   1.76431],
	       [  1.37789,   0.0985 ,   1.28868],
	       [  1.57073,   0.03343,   0.81813],
	       [  1.4647 ,   0.28544,   0.52111],
	       [  1.24719,   0.10138,   0.22746],
	       [  1.56065,   0.21671,   0.19627]]

	values = hcat(values...)

	#
	# Define priors and initial conditions for each model
	#

	#
	# Priors and initial conditions - these are model-specfic as each model can 
	# have different numbers of parameters/species
	#
	priors1 = [Uniform(0.0, 5.0) for i in 1:4]
	ic1 = [20.0, 10.0, 0.0]

	priors2 = vcat([Uniform(0.0, 5.0) for i in 1:4], Uniform(0.0, 10.0))
	ic2 = [20.0, 0.0, 10.0, 0.0]

	priors3 = vcat([Uniform(0.0, 5.0) for i in 1:4], Uniform(0.0, 10.0))
	ic3 = [20.0, 10.0, 0.0]

	modelprior = DiscreteUniform(1, 3)

	# p = (alpha, gamma, d, v)
	# x = (S, I, R)
	function model1(dx, x, p, t)
	    dx[1] = p[1] - p[2]*x[1]*x[2] - p[3]*x[1] # dS/dt = alpha - gamma*S*I - d*S
	    dx[2] = p[3]*x[1]*x[2] - p[4]*x[2] - p[3]*x[2] # dI/dt = gamma*S*I - v*I - d*I
	    dx[3] = p[4]*x[2] - p[3]*x[3] # dR/dt = v*I - d*R
	end

	# p = (alpha, gamma, d, v, delta)
	# x = (S, L, I, R)
	function model2(dx, x, p, t)
	    dx[1] = p[1] - p[2]*x[1]*x[3] - p[3]*x[1] # dS/dt = alpha - gamma*S*I - d*S
	    dx[2] = p[2]*x[1]*x[3] - p[5]*x[2] - p[3]*x[2] # dL/dt = gamma*S*I - delta*L - d*L
	    dx[3] = p[5]*x[2] - p[4]*x[3] - p[3]*x[3] # dI/dt = delta*L - v*I - d*I
	    dx[4] = p[4]*x[3] - p[3]*x[4] # dR/dt = v*I - d*R
	end

	# p = (alpha, gamma, d, v, e)
	# x = (S, I, R)
	function model3(dx, x, p, t)
	    dx[1] = p[1] - p[2]*x[1]*x[2] - p[3]*x[1] + p[5]*x[3] # dS/dt = alpha - gamma*S*I - d*S + e*R
	    dx[2] = p[3]*x[1]*x[2] - p[4]*x[2] - p[3]*x[2] # dI/dt = gamma*S*I - v*I - d*I
	    dx[3] = p[4]*x[2] - (p[3]+p[5])*x[3] # dR/dt = v*I - (d+e)*R
	end

	ics = [ic1, ic2, ic3]

	# Define simulator functions for each model

	simulator1(params) = Array{Float64,2}(
	    solve(ODEProblem(model1, ics[1], (times[1], times[end]), params), saveat=times, force_dtmin=true))

	# Model2 contains the species L, which is not measured - we remove it from the returned ODE solution
	# so that it can be compared to the reference data "values", which only contains S, I and R
	simulator2(params) = Array{Float64,2}(
	    solve(ODEProblem(model2, ics[2], (times[1], times[end]), params), saveat=times, force_dtmin=true))[[1,3,4],:]

	simulator3(params) = Array{Float64,2}(
	    solve(ODEProblem(model3, ics[3], (times[1], times[end]), params), saveat=times, force_dtmin=true))
	
	
	#
	# Do model selection
	#
	input = SimulatedModelSelectionInput(3, n_particles, threshold_schedule, modelprior,
	    [priors1, priors2, priors3], "keep_all", euclidean,
	    [simulator1, simulator2, simulator3], max_iter)

	ms_res = model_selection(input, values);

	#
	# Tests on output shapes
	#
	n_params = [4,5,5]
	@test all(vcat([[size(ms_res.smc_outputs[m].population[i],2) == n_params[m]
		for i in 1:length(threshold_schedule)]
				for m in 1:ms_res.M]...))

	@test all(vcat([[size(ms_res.smc_outputs[m].population[i],1) == ms_res.n_accepted[i][m]
		for i in 1:length(threshold_schedule)]
				for m in 1:ms_res.M]...))

	@test all(vcat([[size(ms_res.smc_outputs[m].distances[i],1) == ms_res.n_accepted[i][m]
		for i in 1:length(threshold_schedule)]
				for m in 1:ms_res.M]...))

	@test all(vcat([[size(ms_res.smc_outputs[m].weights[i],1) == ms_res.n_accepted[i][m]
		for i in 1:length(threshold_schedule)]
				for m in 1:ms_res.M]...))

	@test all([sum(arr) <= n_particles for arr in ms_res.n_accepted])


	@test all([size(ms_res.smc_outputs[m].population,1) == length(threshold_schedule) for m in 1:ms_res.M])
	@test all([size(ms_res.smc_outputs[m].distances,1) == length(threshold_schedule) for m in 1:ms_res.M])
	@test all([size(ms_res.smc_outputs[m].weights,1) == length(threshold_schedule) for m in 1:ms_res.M])

end