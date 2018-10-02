using Base.Test, GpABC, DifferentialEquations, Distances, Distributions

@testset "Model selection test" begin

	threshold_schedule = [20.0, 15.0, 10.0] 
	summary_statistic = "keep_all"
	max_iter = 1e4
	n_particles = 200
	distance_metric = euclidean

	#
	# Experimental data - from ABCSysBio example at
	# https://github.com/jamesscottbrown/abc-sysbio/blob/master/examples/SBML/Example1/SIR.ipynb
	#
	times = [0.0, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
	data = [[ 20.     ,  10.     ,   0.     ],
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

	data = hcat(data...)

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

	modelprior = DiscreteUniform(1, 2)

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

	ics = [ic1, ic2]

	# Define simulator functions for each model

	simulator1(params) = Array{Float64,2}(
	    solve(ODEProblem(model1, ics[1], (times[1], times[end]), params), saveat=times, force_dtmin=true))

	# Model2 contains the species L, which is not measured - we remove it from the returned ODE solution
	# so that it can be compared to the reference data "data", which only contains S, I and R
	simulator2(params) = Array{Float64,2}(
	    solve(ODEProblem(model2, ics[2], (times[1], times[end]), params), saveat=times, force_dtmin=true))[[1,3,4],:]

	#
	# For tests on output shapes
	#
	n_params = [4,5]

	function test_ms_output(ms_res::ModelSelectionOutput, is_simulation::Bool)
		# 2nd dimension of population should always be the number of parameters
		@test all(vcat([[size(ms_res.smc_outputs[m].population[i],2) == n_params[m]
			for i in 1:length(ms_res.threshold_schedule)]
					for m in 1:ms_res.M]...))

		# First dimension of population should always be the number of accepted particles
		@test all(vcat([[size(ms_res.smc_outputs[m].population[i],1) == ms_res.n_accepted[i][m]
			for i in 1:length(ms_res.threshold_schedule)]
				for m in 1:ms_res.M]...))

		# First dimension of distances should always be the number of accepted particles
		@test all(vcat([[size(ms_res.smc_outputs[m].distances[i],1) == ms_res.n_accepted[i][m]
			for i in 1:length(ms_res.threshold_schedule)]
					for m in 1:ms_res.M]...))

		# First dimension of weights should always be the number of accepted particles
		@test all(vcat([[size(ms_res.smc_outputs[m].weights[i],1) == ms_res.n_accepted[i][m]
			for i in 1:length(ms_res.threshold_schedule)]
					for m in 1:ms_res.M]...))

		# Total number of accepted particles at each population can't be more than n_particles 
		@test all([sum(arr) <= n_particles for arr in ms_res.n_accepted])

		# There should be as many populations, distances as weights as populations
		@test all([size(ms_res.smc_outputs[m].population,1) == length(ms_res.threshold_schedule) for m in 1:ms_res.M])
		@test all([size(ms_res.smc_outputs[m].distances,1) == length(ms_res.threshold_schedule) for m in 1:ms_res.M])
		@test all([size(ms_res.smc_outputs[m].weights,1) == length(ms_res.threshold_schedule) for m in 1:ms_res.M])

		# Can't have more than max_iter tries in total at each population - this is only true for simulation
		if is_simulation
			@test all([sum([ms_res.smc_outputs[m].n_tries[i] for m=1:ms_res.M]) <= max_iter for i=1:length(ms_res.threshold_schedule)])
		end

		# Weights must sum to 1 for all models in all cases where the model accepted at least one particle
		weight_sums = vcat([[sum(ms_res.smc_outputs[m].weights[i]) for i = 1:length(ms_res.threshold_schedule)] for m in 1:ms_res.M]...)
		nonzero_weights = vcat([[size(ms_res.smc_outputs[m].weights[i],1) > 0 for i = 1:length(ms_res.threshold_schedule)] for m in 1:ms_res.M]...)
		@test all(weight_sums[nonzero_weights] .== 1.0)
	end


	#
	# Do model selection using simulation
	#
	input = SimulatedModelSelectionInput(2, n_particles, threshold_schedule, modelprior,
	    [priors1, priors2], summary_statistic, distance_metric, [simulator1, simulator2], max_iter)

	ms_res = model_selection(input, data);
	test_ms_output(ms_res, true)

	# User-level function for the same computation
	ms_res  = model_selection(data, n_particles, threshold_schedule, [priors1, priors2],
		summary_statistic, [simulator1, simulator2])
	test_ms_output(ms_res, true)

	#
	# Do model selection using emulation
	#
	n_design_points = 200
	distance_metric = euclidean
	rt = RepetitiveTraining()
	gpkernel = SquaredExponentialArdKernel()

	#
	# A set of functions that return a trained emulator with a prior sampling function as an argument
	#
	emulator_trainers = [f(prior_sampler) = GpABC.abc_train_emulator(prior_sampler,
	        n_design_points,
	        GpABC.build_summary_statistic(summary_statistic)(data),
	        sim,
	        GpABC.build_summary_statistic(summary_statistic),
	        distance_metric)
	    for sim in [simulator1, simulator2]]

    #
	# emulator_settings = [AbcEmulationSettings(n_design_points,
	#         trainer,
	#         (x, em) -> gp_regression(x, em)) for trainer in emulator_trainers]

	input = EmulatedModelSelectionInput(2, 200, threshold_schedule, modelprior, [priors1, priors2],
	    emulator_trainers, 100, 1e3)

	ms_res = model_selection(input, data)
	test_ms_output(ms_res, false)

	# The same computation using user-level function
	model_selection(n_design_points, data, n_particles, threshold_schedule, [priors1, priors2],
		summary_statistic, [simulator1, simulator2])
	test_ms_output(ms_res, false)

	# Repeat above with thresholds that are too small - check for warnings
	ms_res = model_selection(data, n_particles, [0.4, 0.2], [priors1, priors2],
		summary_statistic, [simulator1, simulator2])
	@test !ms_res.completed_all_populations
	ms_res = model_selection(n_design_points, data, n_particles, [0.4, 0.2], [priors1, priors2],
		summary_statistic, [simulator1, simulator2])
	@test !ms_res.completed_all_populations
end