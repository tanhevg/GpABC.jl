#
# Input types
#
abstract type ABCInput end

abstract type ABCRejectionInput <: ABCInput end

"""
    SimulatedABCRejectionInput

An object that defines the settings for a simulation-based rejection-ABC computation.

# Fields
- `n_params::Int64`: The number of parameters to be estimated.
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold::Float64`: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: A 1D Array of distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays.
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
"""
struct SimulatedABCRejectionInput <: ABCRejectionInput
    n_params::Int64
    n_particles::Int64
    threshold::Float64
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
    simulator_function::Function
end

"""
    RepetitiveTrainingSettings

A structure that holds settings for repetitive training of the emulator. On each iteration of re-training a certain number of points is sampled from the prior.
Emulator prediction variance is then obtained for this sample. The points with the highest variance are added to the training set, the model is simulated for
these additional parameters, and the emulator is re-trained.

# Fields
- `rt_iterations`: Number of times the emulator will be re-trained.
- `rt_sample_size`: Size of the sample that will be used to evaluate emulator variance.
- `rt_extra_points`: Number of points to add to the training set at each iteration of re-training. Should not be greater than `rt_sample_size`.
"""
struct RepetitiveTrainingSettings
    rt_iterations::Int64
    rt_sample_size::Int64
    rt_extra_points::Int64
end

RepetitiveTrainingSettings(; rt_iterations::Int64=0, rt_sample_size::Int64=0, rt_extra_points::Int64=0) =
    RepetitiveTrainingSettings(rt_iterations, rt_sample_size, rt_extra_points)

"""
    EmulatedABCRejectionInput

An object that defines the settings for a emulation-based rejection-ABC computation.

# Fields
- `n_params::Int64`: The number of parameters to be estimated.
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold::Float64`: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: A 1D Array of distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `distance_prediction_function::Function`: A function that takes a parameter vector as an argument and outputs a distance to the observed data.
- `batch_size::Int64`: The number of predictions to be made in each batch.
- `max_iter::Int64`: The maximum number of iterations/batches before termination.
"""
struct EmulatedABCRejectionInput <: ABCRejectionInput
	n_params::Int64
	n_particles::Int64
	threshold::Float64
	priors::AbstractArray{ContinuousUnivariateDistribution,1}
	distance_prediction_function::Function
	batch_size::Int64
    max_iter::Int64
    # rt_settings::RepetitiveTrainingSettings
end

# EmulatedABCRejectionInput(n_params::Int64,
#     n_particles::Int64,
#     threshold::Float64,
#     priors::AbstractArray{ContinuousUnivariateDistribution,1},
#     distance_prediction_function::Function,
#     batch_size::Int64,
#     max_iter::Int64) =
# EmulatedABCRejectionInput(n_params,
#     n_particles,
#     threshold,
#     priors,
#     distance_prediction_function,
#     batch_size,
#     max_iter,
#     RepetitiveTrainingSettings())

abstract type ABCSMCInput <: ABCInput end

"""
    SimulatedABCSMCInput

An object that defines the settings for a simulation-based ABC-SMC computation.

# Fields
- `n_params::Int64`: The number of parameters to be estimated.
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: A 1D Array of distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `summary_statistic::Union{String,AbstractArray{String,1},Function}`: Either: 1. A `String` or 1D Array of strings that Or 2. A function that outputs a 1D Array of Floats that summarises model output. REFER TO DOCS
- `distance_function::Function`: Any function that computes the distance between 2 1D Arrays.
- `simulator_function::Function`: A function that takes a parameter vector as an argument and outputs model results.
"""
struct SimulatedABCSMCInput <: ABCSMCInput
    n_params::Int64
    n_particles::Int64
    threshold_schedule::AbstractArray{Float64,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
    simulator_function::Function
end

"""
    EmulatedABCRejectionInput

An object that defines the settings for a emulation-based rejection-ABC computation.

# Fields
- `n_params::Int64`: The number of parameters to be estimated (the length of each parameter vector/particle).
- `n_particles::Int64`: The number of parameter vectors (particles) that will be included in the final posterior.
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `priors::AbstractArray{ContinuousUnivariateDistribution,1}`: A 1D Array of distributions with length `n_params` from which candidate parameter vectors will be sampled.
- `distance_prediction_function::Function`: A function that takes a parameter vector as an argument and outputs a distance to the observed data.
- `batch_size::Int64`: The number of predictions to be made in each batch.
- `max_iter::Int64`: The maximum number of iterations/batches before termination.
"""
struct EmulatedABCSMCInput <: ABCSMCInput
    n_params::Int64
    n_particles::Int64
    threshold_schedule::AbstractArray{Float64,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    distance_prediction_function::Function
    batch_size::Int64
    max_iter::Int64
end

#
# Tracker types
#
abstract type ABCSMCTracker end

mutable struct SimulatedABCSMCTracker <: ABCSMCTracker
    n_params::Int64
    n_accepted::AbstractArray{Int64,1}
    n_tries::Vector{Int64}
    threshold_schedule::AbstractArray{Float64,1}
    population::AbstractArray{AbstractArray{Float64,2},1}
    distances::AbstractArray{AbstractArray{Float64,1},1}
    weights::AbstractArray{StatsBase.Weights,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    summary_statistic::Function
    distance_function::Function
    simulator_function::Function
end

mutable struct EmulatedABCSMCTracker <: ABCSMCTracker
    n_params::Int64
    n_accepted::AbstractArray{Int64,1}
    n_tries::AbstractArray{Int64,1}
    threshold_schedule::AbstractArray{Float64,1}
    population::AbstractArray{AbstractArray{Float64,2},1}
    distances::AbstractArray{AbstractArray{Float64,1},1}
    weights::AbstractArray{StatsBase.Weights,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    distance_prediction_function::Function
    max_iter::Int64
end

#
# Output types
#
abstract type ABCOutput end

"""
    ABCRejectionOutput

A container for the output of a rejection-ABC computation.

# Fields
- `n_params::Int64`: The number of parameters to be estimated.
- `n_accepted::Int64`: The number of accepted parameter vectors (particles) in the posterior.
- `n_tries::Int64`: The total number of parameter vectors (particles) that were tried.
- `threshold::Float64`: The maximum distance from the summarised model output to summarised observed data for a parameter vector to be included in the posterior.
- `population::AbstractArray{Float64,2}`: The parameter vectors (particles) in the posterior. Size: (`n_accepted`, `n_params`).
- `distances::AbstractArray{Float64,1}`: The distances for each parameter vector (particle) in the posterior to the observed data in summary statistic space. Size: (`n_accepted`).
- `weights::StatsBase.Weights`: The weight of each parameter vector (particle) in the posterior.
"""
struct ABCRejectionOutput <: ABCOutput
    n_params::Int64
    n_accepted::Int64
    n_tries::Int64
    threshold::Float64
    population::AbstractArray{Float64,2}
    distances::AbstractArray{Float64,1}
    weights::StatsBase.Weights
end

"""
    ABCSMCOutput

A container for the output of a rejection-ABC computation.

# Fields
- `n_params::Int64`: The number of parameters to be estimated.
- `n_accepted::Int64`: The number of accepted parameter vectors (particles) in the posterior.
- `n_tries::Int64`: The total number of parameter vectors (particles) that were tried.
- `threshold_schedule::AbstractArray{Float64}`: A set of maximum distances from the summarised model output to summarised observed data for a parameter vector to be included in the posterior. Each distance will be used in a single run of the ABC-SMC algorithm.
- `population::AbstractArray{Float64,2}`: The parameter vectors (particles) in the posterior. Size: (`n_accepted`, `n_params`).
- `distances::AbstractArray{Float64,1}`: The distances for each parameter vector (particle) in the posterior to the observed data in summary statistic space. Size: (`n_accepted`).
- `weights::StatsBase.Weights`: The weight of each parameter vector (particle) in the posterior.
"""
struct ABCSMCOutput <: ABCOutput
    n_params::Int64
    n_accepted::AbstractArray{Int64,1}
    n_tries::AbstractArray{Int64,1}
    threshold_schedule::AbstractArray{Float64,1}
    population::AbstractArray{AbstractArray{Float64,2},1}
    distances::AbstractArray{AbstractArray{Float64,1},1}
    weights::AbstractArray{StatsBase.Weights,1}
end

#
# Read/write functions
#
function write_scalar(
        stream::IO,
        x::Real;
        separator="\n",
        )
    write(stream, string(x), separator)
end


function write_vector(
        stream::IO,
        vector::Vector{R};
        separator=" ",
        finalcharacter="\n"
        ) where {
        R<:Real,
        }
    n = length(vector)
    for i in 1:(n - 1)
        write_scalar(stream, vector[i]; separator=separator)
    end
    write_scalar(stream, vector[n]; separator=finalcharacter)
end


function write_matrix(
        stream::IO,
        matrix::Matrix{R};
        separator=" ",
        newline=true,
        ) where {
        R<:Real,
        }
    n = size(matrix, 2)
    for i in 1:(n - 1)
        write_vector(stream, matrix[:, i]; separator=separator, finalcharacter="\n")
    end
    finalcharacter = newline ? "\n" : ""
    write_vector(stream, matrix[:, n]; separator=separator, finalcharacter=finalcharacter)
end


function Base.write(stream::IO, output::ABCRejectionOutput)
    write_scalar(stream, output.n_params)
    write_scalar(stream, output.n_accepted)
    write_scalar(stream, output.n_tries)
    write_scalar(stream, output.threshold)
    write_matrix(stream, output.population)
    write_vector(stream, output.distances)
    write_vector(stream, output.weights.values)

    flush(stream)
end


function Base.write(stream::IO, output::Union{ABCSMCOutput, ABCSMCTracker})
    write_scalar(stream, output.n_params)
    write_vector(stream, output.n_accepted)
    write_vector(stream, output.n_tries)
    write_vector(stream, output.threshold_schedule)
    for i in 1:length(output.threshold_schedule)
        write(stream, "-----\n")
        write_scalar(stream, output.n_accepted[i])
        write_scalar(stream, output.n_tries[i])
        write_scalar(stream, output.threshold_schedule[i])
        write_matrix(stream, output.population[i])
        write_vector(stream, output.distances[i])
        write_vector(stream, output.weights[i].values)
    end

    flush(stream)
end


function write(stream::IO, output::Union{ABCOutput, ABCSMCTracker})
    return Base.write(stream, output)
end


function read_rejection_output(filepath::AbstractString)
    input_file = open(filepath, "r")

    try
        n_params = parse(Int64, readline(input_file))
        n_accepted = parse(Int64, readline(input_file))
        n_tries = parse(Int64, readline(input_file))
        threshold = parse(Float64, readline(input_file))

        population = zeros(n_params, n_accepted)
        for i in 1:n_accepted
            population[:, i] = parse.(Float64, split(readline(input_file)))
        end

        distances = parse.(Float64, split(readline(input_file)))

        wts = parse.(Float64, split(readline(input_file)))
        weights = StatsBase.Weights(wts)

        return ABCRejectionOutput(n_params,
                                  n_accepted,
                                  n_tries,
                                  threshold,
                                  population,
                                  distances,
                                  weights,
                                  )
    finally
        close(input_file)
    end
end


function read_smc_output(filepath::AbstractString)
    input_file = open(filepath, "r")

    try
        n_params = parse(Int64, readline(input_file))
        n_accepted = parse.(Int64, split(readline(input_file)))
        n_tries = parse.(Int64, split(readline(input_file)))
        threshold_schedule = parse.(Float64, split(readline(input_file)))

        @assert length(n_accepted) == length(n_tries) == length(threshold_schedule)

        population = Matrix{Float64}[]
        distances = Vector{Float64}[]
        weights = StatsBase.Weights[]

        for i in 1:length(threshold_schedule)
            separator = readline(input_file)

            @assert parse(Int64, readline(input_file)) == n_accepted[i]
            @assert parse(Int64, readline(input_file)) == n_tries[i]
            @assert parse(Float64, readline(input_file)) == threshold_schedule[i]

            push!(population, zeros(n_params, n_accepted[i]))
            for j in 1:n_accepted[i]
                particle = parse.(Float64, split(readline(input_file)))
                population[i][:, j] = particle
            end

            push!(distances, parse.(Float64, split(readline(input_file))))

            wts = parse.(Float64, split(readline(input_file)))
            push!(weights, StatsBase.Weights(wts))
        end

        return ABCSMCOutput(n_params,
                            n_accepted,
                            n_tries,
                            threshold_schedule,
                            population,
                            distances,
                            weights,
                            )
    finally
        close(input_file)
    end
end

function checkABCInput(input::ABCInput)
	if input.n_params != length(input.priors)
		throw(ArgumentError("There are $(input.n_params) unknown parameters but $(length(input.priors)) priors were provided"))
	end
end
