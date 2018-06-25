"""
test_prior - A no-argument function, returns a sample from test prior, that is
ready to be fed into summary_statistics_function. Returns an array of size (n, d),
n rows and d columns, where n is the number of points,
and d is the dimentionality of the data

summary_statistics_function - A function that takes in a test sample, and returns
a 1D vector of summary statistics. In our case this returns a vector of norms to
the observed data, as emulated by the GP.

observed_summary_statistic - Summary statistic of the observed data. In our case this is zero

See abc_test.jl for usage example
"""
#function rejection_abc(test_prior::Function, summary_statistics_function::Function, epsilon, observed_summary_statistic=0; return_count=-1, max_iterations=1000)
function rejection_abc(test_prior::Function, summary_statistics_function::Function; epsilon=-1, observed_summary_statistic=0, return_count=-1, max_iterations=1000)
    ret_data = nothing
    if epsilon<0
        if return_count>0
            ret_data = rejection_beta_abc_internal(summary_statistics_function, observed_summary_statistic, test_prior, return_count)
        end
    else
        ret_data, _ = rejection_abc_internal(summary_statistics_function, observed_summary_statistic, test_prior, epsilon, return_count, max_iterations)
    end
    ret_data
end

# do not export, use from multiple_training_abc which is exported
function multiple_training_abc_step(summary_statistics_function, observed_summary_statistic, test_prior, epsilon, return_count, max_iterations)
    #in our case they are not new actually when using rejection
    new_test_points_x, new_test_points_weights = rejection_abc_internal(summary_statistics_function, observed_summary_statistic, test_prior, epsilon, return_count, max_iterations)
    new_test_points_cdf = Array{Float64}(size(new_test_points_weights, 1))
    cumsum!(new_test_points_cdf, new_test_points_weights ./ sum(new_test_points_weights))
    return new_test_points_x, new_test_points_weights, new_test_points_cdf
end

# TODO rename to sequential_abc
function smc(init_prior,summary_statistics_function, epsilons,
        testing_points; max_iterations=1000, return_count=-1,observed_summary_statistic=0)

    test_prior = ()-> begin
        rand(init_prior, testing_points)'
    end
    number_of_seq_abc_runs = size(epsilons,1)
    new_test_points_x, new_test_points_weights = rejection_abc_internal(summary_statistics_function, observed_summary_statistic, test_prior, epsilons[1], return_count, max_iterations)
    print(size(new_test_points_x))
    new_test_points_cdf = Array{Float64}(size(new_test_points_weights, 1))
    cumsum!(new_test_points_cdf, new_test_points_weights ./ sum(new_test_points_weights))
    one_k_all_seq_abc_run_accepted = new_test_points_x
    print("1 SMC done.")
    for seq_abc_run_ind in 2:number_of_seq_abc_runs


        vals, vals_weights = importance_sampling_with_mv_normal(init_prior, new_test_points_x,
                new_test_points_weights, new_test_points_cdf, testing_points)
        print(seq_abc_run_ind," importance sampling done.")
        vals_cdf = Array{Float64}(size(vals_weights, 1))
        cumsum!(vals_cdf, vals_weights ./ sum(vals_weights))
        new_prior_func = ()-> begin
                 importance_sampling(vals, vals_cdf,testing_points)
        end
        #new_prior_func = () -> begin
        #    vals, weights = importance_sampling_with_mv_normal(init_prior, new_test_points_x,
        #        new_test_points_weights, new_test_points_cdf, testing_points)
        #    vals
        #end
        new_test_points_x, new_test_points_weights = (rejection_abc_internal(summary_statistics_function, observed_summary_statistic, new_prior_func, epsilons[seq_abc_run_ind], return_count, max_iterations))
        print(seq_abc_run_ind,"rejection ABC done.")
        new_test_points_cdf = Array{Float64}(size(new_test_points_weights, 1))
        cumsum!(new_test_points_cdf, new_test_points_weights ./ sum(new_test_points_weights))
        print("Accepted: ", size(new_test_points_x))
        one_k_all_seq_abc_run_accepted= vcat(one_k_all_seq_abc_run_accepted, new_test_points_x)
    end
    return new_test_points_x, new_test_points_weights, new_test_points_cdf,one_k_all_seq_abc_run_accepted
end


# do not export
function rejection_abc_internal(summary_statistics_function, observed_summary_statistic, test_prior, epsilon, return_count, max_iterations)
    #print(epsilon, return_count, max_iterations)
    data, weights = rejection_abc_internal_step(summary_statistics_function, observed_summary_statistic, test_prior, epsilon)
    if return_count < 0
        return data, weights
    else
        ret_data = Array{Float64}(return_count, size(data, 2))
        ret_weights = Array{Float64}(return_count)
        idx_ret = 1 + copy_batch!(ret_data, ret_weights, data, weights, 1)
        i = 0
        while idx_ret <= return_count && (max_iterations < 0 || i < max_iterations)
            data, weights = rejection_abc_internal_step(summary_statistics_function, observed_summary_statistic, test_prior, epsilon)
            lw = copy_batch!(ret_data, ret_weights, data, weights, idx_ret)
            if lw == 0
                warn("Rejection ABC: got no hits for epsilon=$(epsilon)")
            end
            idx_ret += lw
            i += 1
        end
        return ret_data, ret_weights
    end
end

function copy_batch!(ret_data, ret_weights, data, weights, idx_ret)
    lw = length(weights)
    lr = length(ret_weights)
    remaining_size = lr - idx_ret + 1
    if lw > 0
        if lw <= remaining_size
            ret_data[idx_ret:idx_ret + lw - 1, :] = data
            ret_weights[idx_ret:idx_ret + lw - 1] = weights
        else
            ret_data[idx_ret:lr, :] = data[1:remaining_size, :]
            ret_weights[idx_ret:lr] = weights[1:remaining_size]
            lw = remaining_size
        end
    end
    lw
end

function rejection_abc_internal_step(summary_statistics_function, observed_summary_statistic, test_prior, epsilon)

    test_sample = test_prior()
    #println("test_sample size",size(test_sample))
    summary_output = summary_statistics_function(test_sample)
    weights = abs.(summary_output .- observed_summary_statistic)
    idx = find(weights .< epsilon)
    ret_data = test_sample[idx, :]
    ret_weights = weights[idx]
    return ret_data, ret_weights
end

function rejection_beta_abc_internal(summary_statistics_function, observed_summary_statistic, test_prior, return_count)
    test_sample = test_prior()
    #println("test_sample size",size(test_sample))
    summary_output = summary_statistics_function(test_sample)
    weights = abs.(summary_output .- observed_summary_statistic)
    p=sortperm(weights)
    return test_sample[p[1:return_count],:]
end
