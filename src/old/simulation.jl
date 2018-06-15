
struct SentinelSampleable <: Sampleable{Univariate,Continuous}
end

mutable struct ABC_GP_Emulator
    design_point_count::Int
    design_prior::Sampleable{<:VariateForm, Continuous}
    test_point_count::Int
    test_prior::Sampleable{<:VariateForm, Continuous}

    gpem::AbstractGaussianProcess

    observed_data
    weights
    model_simulator_function::Function
    model_result_summary_statistic_function::Function

    max_attempts::Int
    log_level::Int
    sample_from_posterior::Bool
end

function ABC_GP_Emulator(observed_data, simulator::Function)
    ABC_GP_Emulator(observed_data, simulator, 0, SentinelSampleable(),
        0, SentinelSampleable())
end

function ABC_GP_Emulator(observed_data,
    simulator::Function,
    design_point_count::Int,
    design_prior::Sampleable{<:VariateForm, Continuous},
    test_point_count::Int,
    test_prior::Sampleable{<:VariateForm, Continuous};
    distance_func::Function=euclidean_distance_ode,
    max_attempts=1000,
    log_level=0,
    sample_from_posterior=true)

    gpem = GPModel()

    ABC_GP_Emulator(design_point_count, design_prior,
        test_point_count, test_prior,
        gpem, observed_data, ones(design_point_count), simulator, distance_func,
        max_attempts, log_level, sample_from_posterior)
end

######################################MODEL TOOLS####################################################
function sample_prior_design_points(abc::ABC_GP_Emulator)
    sample_prior_design_points(abc, abc.design_prior, abc.design_point_count)
end

function sample_prior_design_points(abc::ABC_GP_Emulator,
        distr::Sampleable{<:VariateForm, Continuous},
        number_of_models::Int)
    abc.design_prior = distr
    abc.design_point_count = number_of_models
    abc.gpem.gp_training_x = rand(distr, number_of_models)'
end

function sample_prior_test_points(abc::ABC_GP_Emulator)
    sample_prior_test_points(abc, abc.test_prior, abc.test_point_count)
end

function sample_prior_test_points(abc::ABC_GP_Emulator,
        distr::Sampleable{<:VariateForm, Continuous},
        number_of_models::Int)
    abc.test_prior = distr
    abc.test_point_count = number_of_models
    abc.gpem.gp_test_x = rand(distr, number_of_models)'
end

#################################new prior = posterior of run before##################################
#expect problem: n vs. training size
function importance_sampling(abc::ABC_GP_Emulator, posterior_ind,selected_gp_means,posterior_cdf_distr, n, number_of_unknown_params)
    selected_posterior_ind= Array{Int64,1}(n)
    new_points= Array{Float64,2}(number_of_unknown_params,n)
    new_weights= ones(n)
    A=rand(n)
    for a in 1:n
        for i in 1:length(posterior_cdf_distr)
            if A[a]<posterior_cdf_distr[i]
                selected_posterior_ind[a]=posterior_ind[i]
                break
            end
        end
    end
    for i in 1:n
        my = abc.gpem.gp_training_x[:,i]
        d = MvNormal(my, 2*cov(abc.gpem.gp_training_x[:,selected_posterior_ind]'))
        new_point = rand(d)
        sum = 0.1
        for j in 1:n
            xx=0
            for p in 1:number_of_unknown_params
                xx=xx+(new_point[p] - abc.gpem.gp_training_x[p,j]).^2
            end
            sum=sum+abc.weights[j]*sqrt(xx)
        end
        new_weight = pdf(d,new_point)/sum
        new_points[:,i] = new_point
        new_weights[i] = new_weight
    end
    return new_points,new_weights
end

##########################################################################################################

function simulate_models(abc::ABC_GP_Emulator)
    simulate_models(abc, abc.gpem.gp_training_x)
end

function simulate_models(abc::ABC_GP_Emulator, model_parameters)
    if abc.log_level > 0
        println("ABC_GP_Emulator: Starting simulation")
    end
    distances = Array{Float64}(size(model_parameters, 1),1)
    for i in 1:size(distances, 1)
        params = model_parameters[i, :]
        tempSol = abc.model_simulator_function(params)
        distances[i, 1] = abc.model_result_summary_statistic_function(tempSol, abc.observed_data)
    end
    if abc.log_level > 0
        println("ABC_GP_Emulator: Simulation complete")
    end
    abc.gpem.gp_training_y = distances
end

function abc_gp_hyperparam_mle(abc::ABC_GP_Emulator)
    if abc.log_level > 0
        println("ABC_GP_Emulator: Starting MLE estimation of GP hyperparameters")
    end
    gp_train(abc.gpem)
    if abc.log_level > 0
        println("ABC_GP_Emulator: GP hyperparameters estimated")
    end
end

function abc_gp_regression(abc::ABC_GP_Emulator)
    gp_regression(abc.gpem)
end

function abc_emulation(abc::ABC_GP_Emulator)
    sample_prior_design_points(abc)
    simulate_models(abc)
    abc_gp_hyperparam_mle(abc)
    sample_prior_test_points(abc)
    gp_regression(abc.gpem)
end

function abc_reject_emulation(abc::ABC_GP_Emulator, epsilon::Real)
    (abc_mean, abc_cov) = abc_emulation(abc)
    sample_normal = abc.sample_from_posterior ? MvNormal(abc_mean, abc_cov) : nothing
    weights = zeros(abc.test_point_count)
    emulated_data = zeros(abc.test_point_count, size(abc.gpem.gp_test_x, 2))
    i = 1
    count = 0
    while i <= abc.test_point_count && count < abc.max_attempts
        sample = abc.sample_from_posterior ? rand(sample_normal) : abc_mean
        idx = find(sample .< epsilon)
        if i + length(idx) > abc.test_point_count
            idx = idx[1:abc.test_point_count - i + 1]
        end
        emulated_data[i:i + length(idx) - 1, :] = abc.gpem.gp_test_x[idx, :]
        weights[i:i + length(idx) - 1] = epsilon - sample[idx]
        i += length(idx)
        count += 1
    end
    if i < abc.test_point_count
        warn("ABC_GP_Emulator: Sampled only $(i-1) points with distance below $(epsilon) ",
            "out of requested $(abc.test_point_count) after $(abc.max_attempts) attempts.")
    end
    emulated_data, weights
end

"""
Euclidean distance between two vectors of vectors. Suitable for comparing ODE results.
"""
function euclidean_distance_ode(x1::AbstractArray{<:AbstractArray, 1}, x2::AbstractArray{<:AbstractArray, 1})
    x1 = hcat(x1...)
    x2 = hcat(x2...)
    sum([vecnorm(x1[i, :] - x2[i, :]) for i=1:size(x1, 1)])
end

#########################################################EVALUATION TOOLS#############################################################
#1.Test:simulation of test points
#simulate test points and calculate difference to observed data (observedData)
#compare emulated distances (gp_mean) and simulated test distances)
function emulationVsSimulation(abc::ABC_GP_Emulator)
    distances_sim_obs = simulate_models(abc, abc.gpem.gp_test_x)
    return abc.gpem.gp_test_y_mean/distances_sim_obs
end

#TODO why 2:end
#TODO check for <0 now: abs
function reject_abc(gp_mean,epsilon)
    posterior_ind = Array{Int64}(1)
    selected_gp_means = Array{Float64}(1)
    sum=0
    for i in 1:length(gp_mean)
        if abs(gp_mean[i])<epsilon
            sum=sum + (epsilon-abs(gp_mean[i]))
            push!(posterior_ind, i)
            push!(selected_gp_means, epsilon-abs(gp_mean[i]))
        end
    end
    posterior_cdf_distr= selected_gp_means/sum
    for i in 2:size(posterior_cdf_distr,1)
        posterior_cdf_distr[i]=posterior_cdf_distr[i-1]+posterior_cdf_distr[i]
        #println(i," and ",posterior_cdf_distr[i])
    end
    return posterior_ind[2:end],selected_gp_means[2:end],posterior_cdf_distr[2:end]
end

#####################################################################################################################################
