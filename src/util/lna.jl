    #this function is the mean variance decomposition of the LNAS
    ##LNA_Mean_Var takes the inputs and then performs the simulating of the mean (ODE solution) and the covariance matrix which changes over time.
    ##x0 is the initial conditions of the DE and the covariance matrix
    ##S is the stochiometry matrix
    ##make_f is a function of the rates - MUST be in the same order as the stochiometry matrix is laid out
    ##volume is the volume of reactants
    #
    #
    #
    #the maths for this can be found in UserManual_v28102013.pdf in the LNA folder in Gaussian Procceses Drop box, p2 equation (7) - the ODE of the covariance. A volume term has been added which can be seen in other sources - the particular source was linked as the layout is the similar to the one implemented.
lna_covariance = function(params::AbstractArray{Float64,1},Tspan::Tuple{Float64,Float64},
        x0::AbstractArray{Float64}, solver::DEAlgorithm,
        saveat::Float64, S::AbstractArray{Float64,2},
        make_f::Function, volume::Float64)

    if volume <= 0.0
        throw(ArgumentError("To use LNA please provide a positive volume of the
            reactants"))
    end

    no_of_species, no_of_reactions = size(S)

    function Mean_ODE(dx, x, pars, t)
        D = ForwardDiff.jacobian(y -> make_f(y, pars), diag(x))
        D = D[:,1:no_of_species]
        A = S*D
        dx[1:no_of_species, 1:no_of_species] = diagm(S*make_f(diag(x),pars))
        dx[no_of_species+1:end,1:no_of_species]= A*x[no_of_species+1:no_of_species*2,1:no_of_species] + x[no_of_species+1:no_of_species*2,1:no_of_species]*(A') + (1/sqrt(volume))*S* diagm(make_f(diag(x),pars)) * S'
    end

    prob = ODEProblem(Mean_ODE, x0, Tspan, params)
    mean_and_var = solve(prob, solver, saveat=saveat)

    mean_traj = Array{Float64,2}(no_of_species, length(mean_and_var.t))
    covar_traj = Array{Array{Float64,2},1}(length(mean_and_var.t))

    for j in 1:length(mean_and_var.t)
        mean_traj[:,j] = diag(mean_and_var[j][1:no_of_species,1:no_of_species])
        covar_traj[j] = mean_and_var[j][no_of_species+1:end, 1:no_of_species]
    end

    return mean_traj, covar_traj, mean_and_var.t

end
