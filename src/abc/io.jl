#
# Input types
#
abstract type ABCInput end

abstract type ABCRejectionInput <: ABCInput end

struct SimulatedABCRejectionInput <: ABCRejectionInput
    n_params::Int64
    n_particles::Int64
    threshold::Float64
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
    simulator_function::Function
end

struct EmulatedABCRejectionInput <: ABCRejectionInput
	n_params::Int64
	n_particles::Int64
	threshold::Float64
	priors::AbstractArray{ContinuousUnivariateDistribution,1}
	distance_prediction_function::Function
	batch_size::Int64
    max_iter::Int
end

abstract type ABCSMCInput <: ABCInput end

struct SimulatedABCSMCInput <: ABCSMCInput
    n_params::Int64
    n_particles::Int64
    threshold_schedule::AbstractArray{Float64,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    summary_statistic::Union{String,AbstractArray{String,1},Function}
    distance_function::Function
    simulator_function::Function
end

struct EmulatedABCSMCInput <: ABCSMCInput
    n_params::Int64
    n_particles::Int64
    threshold_schedule::AbstractArray{Float64,1}
    priors::AbstractArray{ContinuousUnivariateDistribution,1}
    distance_prediction_function::Function
    batch_size::Int64
    max_iter::Int
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
    max_iter::Int
end

#
# Output types
#
abstract type ABCOutput end

struct ABCRejectionOutput <: ABCOutput
    n_params::Int64
    n_accepted::Int64
    n_tries::Int64
    threshold::Float64
    population::AbstractArray{Float64,2}
    distances::AbstractArray{Float64,1}
    weights::StatsBase.Weights
end


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
