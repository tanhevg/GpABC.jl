struct LNAInput
    params::AbstractArray{Float64,1}
    S::AbstractArray{Float64,2}
    reaction_rate_function::Function
    volume::Float64
end

struct LNA
    traj_means::AbstractArray{Float64,2}
    traj_covars::AbstractArray{AbstractArray{Float64,2},1}
    time_points::AbstractArray{Float64,1}
end

function compute_LNA(input::LNAInput,
    x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
    Tspan::Tuple{Float64,Float64},
    saveat::Float64,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();
    kwargs...)

    if input.volume <= 0.0
        throw(ArgumentError("To use LNA please provide a positive volume of the
            reactants"))
    end

    no_of_species, no_of_reactions = size(input.S)

    function Mean_ODE(dx, x, pars, t)
        D = ForwardDiff.jacobian(y -> input.reaction_rate_function(y, pars), diag(x))
        D = D[:,1:no_of_species]
        A = input.S*D
        dx[1:no_of_species, 1:no_of_species] = diagm(input.S*input.reaction_rate_function(diag(x),pars))
        dx[no_of_species+1:end,1:no_of_species]= A*x[no_of_species+1:no_of_species*2,1:no_of_species] + x[no_of_species+1:no_of_species*2,1:no_of_species]*(A') + (1/sqrt(input.volume))*input.S* diagm(input.reaction_rate_function(diag(x),pars)) * input.S'
    end

    prob = ODEProblem(Mean_ODE, vcat(diagm(x0[1]), x0[2]), Tspan, input.params)
    mean_and_var = solve(prob, solver, saveat=saveat; kwargs...)

    mean_traj = Array{Float64,2}(no_of_species, length(mean_and_var.t))
    covar_traj = Array{Array{Float64,2},1}(length(mean_and_var.t))

    for j in 1:length(mean_and_var.t)
        mean_traj[:,j] = diag(mean_and_var[j][1:no_of_species,1:no_of_species])
        covar_traj[j] = mean_and_var[j][no_of_species+1:end, 1:no_of_species]
    end

    return LNA(mean_traj, covar_traj, mean_and_var.t)
end

function sample_LNA_trajectories(lna::LNA, n_samples::Int64)
    trajectories = [rand(MvNormal(lna.traj_means[:,j], lna.traj_covars[j]), n_samples) for j in 1:length(lna.time_points)]
    return hcat([mean(traj,2) for traj in trajectories]...)
end

function get_LNA_trajectories(input::LNAInput, n_samples::Int64,
    x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
    Tspan::Tuple{Float64,Float64},
    saveat::Float64,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();
    kwargs...)

    lna = compute_LNA(input, x0, Tspan, saveat, solver; kwargs...)
    return sample_LNA_trajectories(lna, n_samples)
end



    #this function is the mean variance decomposition of the LNAS
    ##LNA_Mean_Var takes the inputs and then performs the simulating of the mean (ODE solution) and the covariance matrix which changes over time.
    ##x0 is the initial conditions of the DE and the covariance matrix
    ##S is the stochiometry matrix
    ##reaction_rate_function is a function of the rates - MUST be in the same order as the stochiometry matrix is laid out
    ##volume is the volume of reactants
    #
    #
    #
    #the maths for this can be found in UserManual_v28102013.pdf in the LNA folder in Gaussian Procceses Drop box, p2 equation (7) - the ODE of the covariance. A volume term has been added which can be seen in other sources - the particular source was linked as the layout is the similar to the one implemented.
"""
    lna_covariance(params::AbstractArray{Float64,1},Tspan::Tuple{Float64,Float64},
            x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
            solver::DEAlgorithm,
            saveat::Float64, S::AbstractArray{Float64,2},
            reaction_rate_function::Function, volume::Float64)

Prose description

# Arguments
- `params::AbstractArray{Float64,1}`: The rate parameters of the stochastic model.
- `Tspan::Tuple{Float64,Float64}`: The start and end times of the simulation.
- `x0::Tuple{AbstractArray{Float64,2},AbstractArray{Float64,2}}`: The initial conditions of the system. Size: number of species x 2*(number of species). Intial number of species x number of species block being a diagonal matrix with the initial conditions of your variables, in the same order as the stochiometry matrix. The second number of species x number of species block is the initial covariance matrix of the system.
- `solver::DEAlgorithm`: The ODE solver the user wishes to use, for example DifferentialEquations.RK4() .
- `saveat::Float64`: The number of time points the use wishes to solve the system for.
- `S::AbstractArray{Float64,2}`: the stochiometry matrix of the system. Size: number of reactions x number of species.
- `reaction_rate_function::Function,`: This is a function f(x, parameters) which should return an array of the reaction rates of the system, i.e. S*f would describe the ODE representation of the system.
- `volume::Float64`: The volume of the reactants of the system.

# Return
- `mean_traj`: A (number of species) x (number of time points) array which holds the mean trajectory for each species on each row of the array.
- `covar_traj`: An array which holds the covariance matrix of the species at each time point.
- `mean_and_var.t`: The timepoints the system was solved for.

If the user wishes to sample trajectories they must sample from a multivariate normal at each time point, t_i which belongs to `mean_and_var.t`. In other words sampling from a multivariate Normal Distribution with mean: `mean_traj[t_i]` and covariance: `covar_traj[t_i]`.
"""
