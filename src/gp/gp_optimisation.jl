
function check_hypers_size(gpem::GPModel,
        hp_lower::AbstractArray{Float64, 1}, hp_upper::AbstractArray{Float64, 1})
    expected_hypers_size = get_hyperparameters_size(gpem)
    if length(hp_lower) > 0 && length(hp_lower) != expected_hypers_size
        error("Incorrect size of hyperparameters vector lower bound for ",
            "$(typeof(gpem.kernel)): $(length(hp_lower)). ",
            "Expected $(expected_hypers_size).")
    end
    if length(hp_upper) > 0 && length(hp_upper) != expected_hypers_size
        error("Incorrect size of hyperparameters vector upper bound for ",
            "$(typeof(gpem.kernel)): $(length(hp_upper)). ",
            "Expected $(expected_hypers_size).")
    end
end

"""
    gp_train(gpm::GPModel; <optional keyword arguments>)

Find Maximum Likelihood Estimate of Gaussian Process hyperparameters by maximising
[`gp_loglikelihood`](@ref), using [`Optim`](http://julianlsolvers.github.io/Optim.jl/stable/) package. The optimisation target is [`gp_loglikelihood_log`](@ref), with gradient computed by [`gp_loglikelihood_grad`](@ref). Internally, this function optimises the MLE with respect to logarithms of hyperparameters. This is done for numerical stability. Logarithmisation and exponentiation is performed by this funtion, i.e. real hyperparameters, not logarithms, are taken in and returned back.

By default,
[Conjugate Gradient](http://julianlsolvers.github.io/Optim.jl/stable/algo/cg/) bounded box optimisation is used, as long as the gradient
with respect to hyperparameters ([`covariance_grad`](@ref)) is implemented for the kernel function. If the gradient
implementation is not provided, [Nelder Mead](http://julianlsolvers.github.io/Optim.jl/stable/algo/nelder_mead/) optimiser is used by default.

# Mandatory argument
- `gpm`: the [`GPModel`](@ref), that contains the training data (x and y),
  the kernel and the starting hyperparameters that will be used for optimisation.

# Optional keyword arguments
- `optimiser::Type{<:Optim.AbstractOptimizer}`: the solver to use.
  If not given, then `ConjugateGradient` will be used for kernels that have gradient
  implementation, and `NelderMead` will be used for those that don't.
- `hp_lower::AbstractArray{Float64, 1}`: the lower boundary for box optimisation.
  Defaults to ``e^{-10}`` for all hyperparameters.
- `hp_upper::AbstractArray{Float64, 1}`: the upper boundary for box optimisation.
  Defaults to ``e^{10}`` for all hyperparameters.
- `log_level::Int`: log level. Default is `0`, which is no logging at all. `1`
  makes `gp_train` print basic information to standard output. `2` switches `Optim`
  logging on, in addition to `1`.

# Return
The list of all hyperparameters, including the standard deviation of the measurement
noise ``\\sigma_n``. Note that after this function returns, the hyperparameters of `gpm`
will be set to the optimised value, and there is no need to call [`set_hyperparameters`](@ref)
once again.
"""
function gp_train(gpem::GPModel;
        optimiser::Union{Nothing, TOpt}=nothing,
        hp_lower::AbstractArray{Float64, 1}=zeros(0),
        hp_upper::AbstractArray{Float64, 1}=zeros(0),
        log_level::Int=0) where {TOpt<:Optim.AbstractOptimizer}
    check_hypers_size(gpem, hp_lower, hp_upper)
    hypers = log.(gpem.gp_hyperparameters)
    kernel_hypers = hypers[1:end-1]
    test_grad_result = covariance_grad(gpem.kernel,
        kernel_hypers, gpem.gp_training_x,
        zeros(size(gpem.gp_training_x, 1), size(gpem.gp_training_x, 1)))
    show_trace = log_level >= 2
    opt_res = nothing
    if test_grad_result === :covariance_gradient_not_implemented
        if optimiser === nothing
            optimiser = NelderMead()
        end
        if log_level > 0
            println("Unbound optimisation using $(typeof(optimiser)). ",
            "No gradient provided. Start point: $(hypers).")
        end
        try
            opt_res = optimize(theta->-gp_loglikelihood_log(theta, gpem),
                hypers,
                optimiser)
        catch ex
            last_hypers = gpem.cache.theta
            gpem.gp_hyperparameters = exp.(last_hypers)
            println("Exception ", ex, " for hyperparameters $(gpem.gp_hyperparameters)")
            rethrow(ex)
        end
    else
        if optimiser === nothing
            optimiser = ConjugateGradient()
        end
        expected_hypers_size = get_hyperparameters_size(gpem)
        if length(hp_lower) == 0
            hp_lower = fill(-10.0, expected_hypers_size)
        else
            hp_lower = log.(hp_lower)
        end
        if length(hp_upper) == 0
            hp_upper = fill(10.0, expected_hypers_size)
        else
            hp_upper = log.(hp_upper)
        end

        if log_level > 0
            println("Bound optimisation using $(typeof(optimiser)). ",
            "Gradient provided. Start point: $(gpem.gp_hyperparameters); ",
            "lower bound: $(hp_lower); upper bound: $(hp_upper)")
        end
        try
            opt_res = optimize(theta->-gp_loglikelihood_log(theta, gpem),
                (storage, theta) -> storage[:] = -gp_loglikelihood_grad(theta, gpem),
                hp_lower, hp_upper,
                hypers,
                Fminbox(optimiser),
                Optim.Options(show_trace = show_trace))
        catch ex
            last_hypers = gpem.cache.theta
            gpem.gp_hyperparameters = exp.(last_hypers)
            println("Exception ", ex, " for hyperparameters $(last_hypers)")
            rethrow(ex)
        end
    end
    if log_level > 0
         println(opt_res)
         println("Optimized log hyperparameters: ", opt_res.minimizer)
         println("Optimized hyperparameters: ", exp.(opt_res.minimizer))
    end
    gpem.gp_hyperparameters = exp.(opt_res.minimizer)
    return exp.(opt_res.minimizer)
end;
