"""
    LNAInput

This is a structure which holds the inputs needed for computing the Linear Noise Approximation (LNA). This structure will hold the stochastic system as provided by the user; uniquely defined through kinetic parameters, the rates of the system and the stoichiometry matrix.

# Arguments
- `params::AbstractArray{Float64,1}`: The rate parameters of the stochastic model.
- `S::AbstractArray{Float64,2}`: the stochiometry matrix of the system. Size: number of reactions x number of species.
- `reaction_rate_function::Function,`: This is a function f(x, parameters) which should return an array of the reaction rates of the system, i.e. S*f would describe the ODE representation of the system.
- `volume::Float64`: The volume of the reactants of the system.
"""

struct LNAInput
    params::AbstractArray{Float64,1}
    S::AbstractArray{Float64,2}
    reaction_rate_function::Function
    volume::Float64
end


"""
    LNA

This is a structure which will hold the LNA: the mean of the trajectories and the covariance between the species.

# Arguments
- `traj_means`: A (number of species) x (number of time points) array which holds the mean trajectory for each species on each row of the array.
- `traj_covars`: An array which holds the covariance matrix of the species at each time point.
- `time_points`: The timepoints the system was solved for.
"""


struct LNA
    traj_means::AbstractArray{Float64,2}
    traj_covars::AbstractArray{AbstractArray{Float64,2},1}
    time_points::AbstractArray{Float64,1}
end

"""
    compute_LNA(input::LNAInput,
        x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
        Tspan::Tuple{Float64,Float64},
        saveat::Float64,
        solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();
        kwargs...)

The function computes the linear noise approximation to system through construction of two ODEs: one describing the trajectories of the mean of the LNA and the other describing the change the covariance between the variables. These outputs are held in a LNA structure.

# Arguments
- `x0::Tuple{AbstractArray{Float64,2},AbstractArray{Float64,2}}`: The initial conditions of the system. In the form of (the initial conditions of the species, the initial covariance matrix of the system).
- `Tspan::Tuple{Float64,Float64}`: The start and end times of the simulation.
- `saveat::Float64`: The number of time points the use wishes to solve the system for.
- `solver::DEAlgorithm`: The ODE solver the user wishes to use, for example `DifferentialEquations.RK4()`.

#Returns
- LNA
"""

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


    ### the below ODE solves the ODE of the mean and the ODE of the covariance together. 

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

"""
    sample_LNA_trajectories(lna::LNA, n_samples::Int64)

A function which samples from the LNA to output sampled trajectories. The LNA gives the mean of the tracjectories and the covariance between them; hence a single trajectory can be sampled from a Multivariate Normal distribution. The user can also sample more than one trajectory; which are then averaged.

# Arguments
- `lna::LNA`: LNA stucture.
- `n_samples::Int64`: The number of sampled tracjectories to be sampled and then averaged.

#Returns
-  A (number of species) x (number of time points) array which holds the averaged trajectory for each species on each row of the array.
"""

function sample_LNA_trajectories(lna::LNA, n_samples::Int64)
    trajectories = [rand(MvNormal(lna.traj_means[:,j], lna.traj_covars[j]), n_samples) for j in 1:length(lna.time_points)]
    return hcat([mean(traj,2) for traj in trajectories]...)
end

"""
    get_LNA_trajectories(input::LNAInput, n_samples::Int64,
        x0::Tuple{AbstractArray{Float64,1},AbstractArray{Float64,2}},
        Tspan::Tuple{Float64,Float64},
        saveat::Float64,
        solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=RK4();
        kwargs...)


A function which computes the LNA and then samples from the it to output sampled trajectories. The user can also sample more than one trajectory; which are then averaged.


# Arguments
- `input::LNAInput`: LNAInput stucture.
- `n_samples::Int64`: The number of sampled tracjectories to be sampled and then averaged.
- `x0::Tuple{AbstractArray{Float64,2},AbstractArray{Float64,2}}`: The initial conditions of the system. In the form of (the initial conditions of the species, the initial covariance matrix of the system).
- `Tspan::Tuple{Float64,Float64}`: The start and end times of the simulation.
- `saveat::Float64`: The number of time points the use wishes to solve the system for.
- `solver::DEAlgorithm`: The ODE solver the user wishes to use, for example DifferentialEquations.RK4() .

#Returns
-  A (number of species) x (number of time points) array which holds the averaged trajectory for each species on each row of the array.
"""

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
