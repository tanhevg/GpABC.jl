#
# Create functions if a string is passed as summary_statistic
#
function build_summary_statistic(summary_statistic_string::String)

    if summary_statistic_string == "mean"
        return mean_summary_statistic
    elseif summary_statistic_string == "variance"
        return var_summary_statistic
    elseif summary_statistic_string == "max"
        return max_summary_statistic
    elseif summary_statistic_string == "min"
        return min_summary_statistic
    elseif summary_statistic_string == "range"
        return range_summary_statistic
    elseif summary_statistic_string == "median"
        return median_summary_statistic
    elseif summary_statistic_string == "q1"
        return q1_summary_statistic
    elseif summary_statistic_string == "q3"
        return q3_summary_statistic
    elseif summary_statistic_string == "iqr"
        return iqr_summary_statistic
    elseif summary_statistic_string == "keep_all"
        return keep_all_summary_statistic
    else
        error("$(summary_statistic_string) is not a valid summary statistic")
    end

end

function build_summary_statistic(summary_statistic_strings::Vector{String})
    function summary_statistic(data)
        sum_stat_funcs = [build_summary_statistic(sum_stat) for sum_stat in summary_statistic_strings]
        out = [func(data) for func in sum_stat_funcs]
        return vcat(out...)
    end
    return summary_statistic
end

#
# Check a user-defined summary statistic function
#
function build_summary_statistic(summary_statistic_func::Function)
    # Check output shape is right
    test_output = summary_statistic_func(rand(3,100))

    if ndims(test_output) != 1
        error("The summary statistic function provided does not return a 1D array")
    end

    return summary_statistic_func
end

#
# Built-in summary statistics. Data should be in the format returned by
# DifferentialEquations.solve i.e. an array with size (n_trajectories, n_time_points)
#

# Mean of each trajectory
function mean_summary_statistic(data::AbstractArray{Float64,2})
    return mean(data, dims=2)[:]
end

# Variance of each trajectory
function var_summary_statistic(data::AbstractArray{Float64,2})
    return var(data, dims=2)[:]
end

# Max of each trajectory
function max_summary_statistic(data::AbstractArray{Float64,2})
    return maximum(data, dims=2)[:]
end

# Min of each trajectory
function min_summary_statistic(data::AbstractArray{Float64,2})
    return minimum(data, dims=2)[:]
end

# Range of each trajectory
function range_summary_statistic(data::AbstractArray{Float64,2})
    return max_summary_statistic(data) - min_summary_statistic(data)
end

# Median of each trajectory
function median_summary_statistic(data::AbstractArray{Float64,2})
    return median(data, dims=2)[:]
end

# First quartile of each trajectory
function q1_summary_statistic(data::AbstractArray{Float64,2})
    return [quantile(data[i,:], 0.25) for i = 1:size(data,1)]
end

# Third quartile of each trajectory
function q3_summary_statistic(data::AbstractArray{Float64,2})
    return [quantile(data[i,:], 0.75) for i = 1:size(data,1)]
end

# Inter-quartile range of each trajectory
function iqr_summary_statistic(data::AbstractArray{Float64,2})
    return q3_summary_statistic(data) - q1_summary_statistic(data)
end

# Keep all the data from all the trajectoriess (in a single 1d array)
function keep_all_summary_statistic(data::AbstractArray{Float64,2})
    return reshape(data, prod(size(data)))
end
