
function euclidean_distance_ode(x1::AbstractArray{<:AbstractArray, 1}, x2::AbstractArray{<:AbstractArray, 1})
    x1 = hcat(x1...)
    x2 = hcat(x2...)
    sum([vecnorm(x1[i, :] - x2[i, :]) for i=1:size(x1, 1)])
end

function importance_sampling(xes, cdfs,return_count)
    n = size(xes,1)
    number_of_unknown_params = size(xes,2)
    selected_gp_training_x = Array{Float64,2}(return_count,number_of_unknown_params)
    A = rand(return_count)
    for a in 1:return_count
        for i in 1:n
            if A[a]<cdfs[i]
                selected_gp_training_x[a,:]=xes[i,:]
                break
            end
        end
    end
    return selected_gp_training_x
end

function get_head_of_ref_table(all_xes, all_weights,return_count)
    p=sortperm(all_weights, rev=true)
    return all_xes[p[1:return_count],:]
end

function multiple_training_abc(gp_training_x, gp_training_y, K, epsilon, return_count,  static_params, simulate_function::Function, summary_statistics_function::Function, observed_data, observed_summary_statistic,max_iterations)
    init_train_size = size(gp_training_x,1)
    number_of_unknown_params = size(gp_training_x,2)
    all_weights = ones(init_train_size)
    for i in 1:K
            println("Starting ",i,"th big circle...")
            println("\t Starting training of", i, "th circle...")
            gpem = GPModel(gp_training_x, gp_training_y)
            gp_train(gpem, log_level=1)
            println("\t\t ...training of ", i, "th circle done.")
            #per iteration the training set size increases by return count

            new_xes = Array{Float64, 2}
            new_estimated_weights = ones(return_count)
            new_weights = ones(return_count)
            new_yes = Array{Float64, 1}
            #returns testing prior
            prior_func() = get_head_of_ref_table(gpem.gp_training_x, all_weights, return_count)



            println("\t Starting abc of", i, "th circle...")
            #performs abc and returns weighted xes with cdf
            #these weights are based on emulation yes!!!!!!!!!!!!
            #in our case they are not new actually when using rejection

            new_test_points_x, new_test_points_weights, new_test_points_cdf = multiple_training_abc_step(
                (x) -> summary_statistics_function(x, gpem),
                observed_summary_statistic, prior_func, epsilon, return_count, max_iterations)

            println("\t\t ...abc of", i, "th circle done.")
            #println("\n\nnew_test_points_x\n ",new_test_points_x,"\nnew_test_points_weights\n", new_test_points_weights, "\nnew_test_points_cdf\n",new_test_points_cdf,"\n\n" )


            println("\t Starting importance sampling of", i, "th circle...")
            new_xes, new_estimated_weights = importance_sampling_with_mv_normal(new_test_points_x, new_test_points_weights, new_test_points_cdf)
            println("\t\t ...importance sampling of", i, "th circle done.")
            #println("\n\new_xes\n ",new_xes,"\new_estimated_weights\n", new_estimated_weights,"\n\n" )


            println("\t Starting  of simulating", i, "th circle...")
            new_yes = simulate_function(new_xes)
            append!(all_weights, new_estimated_weights)

            println("\t\t simulating of ", i, "th circle done.")
            gp_training_x = vcat(gp_training_x, new_xes)
            gp_training_y = vcat(gp_training_y, new_yes)

            println("...", i, "th big circle done.")
            #println("RESULT of ",i,"th training:")
            #println("new_estimated_weights ",new_estimated_weights)
            #println("gp_training_x ",gp_training_x)
            #println("gp_training_y  ",gp_training_y )
    end
    println("ALL DONE! YOU GOT DISSSS!")
    return gp_training_x, gp_training_y, all_weights
end

function multiple_training_seq_abc(gp_training_x, gp_training_y, K, epsilons, return_count,
                                static_params, simulate_function::Function, summary_statistics_function::Function,
                                observed_data, observed_summary_statistic,max_iterations)

    init_train_size = size(gp_training_x,1)
    testing_points = init_train_size
    number_of_unknown_params = size(gp_training_x,2)
    all_weights = ones(init_train_size)
    all_k_all_seq_abc_run_accepted=[]
    all_thetas =[]
    for i in 1:K
            gpem = GPModel(gp_training_x, gp_training_y)
            theta=(gp_train(gpem))
            println("\t\t theta is ",theta)
            new_xes = Array{Float64, 2}
            new_estimated_weights = ones(return_count)
            prior_func() = get_head_of_ref_table(gpem.gp_training_x, all_weights, return_count)
            #if error check args order
            #problem: init_prior vs prior_func
            #will not work
            new_test_points_x, new_test_points_weights, new_test_points_cdf,one_k_all_seq_abc_run_accepted = smc(init_prior,
                (x) -> summary_statistics_function(x, gpem),
                prior_func, epsilons, testing_points, max_iterations, return_count, observed_summary_statistic)
            new_xes, new_estimated_weights = importance_sampling_with_mv_normal(init_prior,
                new_test_points_x, new_test_points_weights, new_test_points_cdf)
            new_yes = simulate_function(new_xes)
            append!(all_weights, new_estimated_weights)
            all_k_all_seq_abc_run_accepted= (all_k_all_seq_abc_run_accepted,one_k_all_seq_abc_run_accepted)
            all_thetas =(all_thetas,theta)
            gp_training_x = vcat(gp_training_x, new_xes)
            gp_training_y = vcat(gp_training_y, new_yes)
    end
    return gp_training_x, gp_training_y, all_weights,all_k_all_seq_abc_run_accepted,all_thetas
end

function importance_sampling_with_mv_normal(init_prior,new_test_points_x, new_test_points_weights, new_test_points_cdf,n=-1)
    if n < 0
        n = size(new_test_points_x, 1)
    end
    number_of_unknown_params = size(new_test_points_x, 2)
    template_points = importance_sampling(new_test_points_x, new_test_points_cdf, n)
    new_training_xes = Array{Float64,2}(n, number_of_unknown_params)
    new_weights = ones(n)
    mx = 2*cov(new_test_points_x) + I * 1e-6
    if any(isnan.(mx))
        println("NaN in covariance matrix")
        mx = ones(length(temp_template_point))
    end
    counter =0
    for i in 1:n
        temp_template_point = template_points[i,:]
        d = nothing
        try
            d = MvNormal(temp_template_point, mx)
        catch ex
            println(ex)
            println(mx)
            throw(ex)
        end
        new_training_x = rand(d)

        while pdf(init_prior,new_training_x)==0
            counter=counter+1
            new_training_x = rand(d)
        end
        dist = sum([vecnorm(new_training_x[:] - new_test_points_x[i, :]) for i=1:size(new_test_points_x, 1)])
        new_weight = pdf(init_prior,new_training_x)/dist
        new_training_xes[i,:] = new_training_x
        new_weights[i] = new_weight
    end
    print(counter, "counter")
    return new_training_xes, new_weights
end
