module LNA

    using ForwardDiff, DifferentialEquations

    #LNAdecomp is the function Fei wants to include

    ##LNAdecomp takes the inputs and then performs the simulating of an LNA trajectory.
    ##x0 is the initial conditions of the DE and the noise terms
    ##S is the stochiometry matrix
    ##make_f is a function of the rates - MUST be in the same order as the stochiometry matrix is laid out
    ##volume is the volume of reactants
    #
    #
    #
    # the maths behind this function can be found on BayesianInference_LNA.pdf in the LNA dropbox p3, equations (1) and (3).
    LNAdecomp = function(params::AbstractArray{Float64,1},Tspan::Tuple{Float64,Float64},
        x0::AbstractArray{Float64}, solver::StochasticDiffEq.StochasticDiffEqAlgorithm,
        dt::Float64,S::AbstractArray{Float64,2},
        make_f::Function, volume::Float64)

        if volume <= 0.0
            throw(ArgumentError("To use LNA please provide a positive volume of the
                reactants"))
        end

        no_of_species, no_of_reactions=size(S)
        function ODE(t,x,dx)
            D=ForwardDiff.jacobian(y -> make_f(y, params),x)
            D=D[:,1:no_of_species]
            A=S*D
            dx[1:no_of_species]= S*make_f(x,params)
            dx[no_of_species+1:end]=  A*x[no_of_species+1:no_of_species*2]
        end
        E_null= zeros(no_of_species,no_of_reactions)
        E_full=vcat(E_null, S)
        noise = zeros(no_of_species*2,no_of_reactions)
        function SDE(t,x,dx)
            #for i in 1:no_of_species*2
                #for j in 1:no_of_reactions
                    #dx[i,j]=(E_full[i,j] * sqrt(abs.(make_f(x,params)[j])))
                #end
            #end

            ## Fei's additions/corrections below
            noise[1:no_of_species*2,1:no_of_reactions] = E_full.*(ones(no_of_species*2,1)*sqrt.(abs.(make_f(x,params)))')
            dx[1:no_of_species*2,1:no_of_reactions] = E_full.*(ones(no_of_species*2,1)*sqrt.(abs.(make_f(x,params)))')

        end

        prob = SDEProblem(ODE,SDE,x0,Tspan,noise_rate_prototype=zeros(no_of_species*2, no_of_reactions))

        sols=solve(prob, solver,  dt=dt)

        LNA=[]
        for i in 1:length(sols)
            push!(LNA,sols[i][1:no_of_species] + (volume)^(-0.5)*sols[i][no_of_species+1:end])
        end

        return LNA, sols.t


    end

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
    LNA_Mean_Var = function(params::AbstractArray{Float64,1},Tspan::Tuple{Float64,Float64},
            x0::AbstractArray{Float64}, solver::DEAlgorithm,
            saveat::Float64, S::AbstractArray{Float64,2},
            make_f::Function, volume::Float64)

        if volume <= 0.0
            throw(ArgumentError("To use LNA please provide a positive volume of the
                reactants"))
        end

        no_of_species, no_of_reactions=size(S)

        function Mean_ODE(t,x,dx)
            D=ForwardDiff.jacobian(y -> make_f(y, params), diag(x))
            D=D[:,1:no_of_species]
            A=S*D
            dx[1:no_of_species, 1:no_of_species] = diagm(S*make_f(diag(x),params))
            dx[no_of_species+1:end,1:no_of_species]= A*x[no_of_species+1:no_of_species*2,1:no_of_species] + x[no_of_species+1:no_of_species*2,1:no_of_species]*(A') + (1/sqrt(volume))*S* diagm(make_f(diag(x),params)) * S'
        end

        prob = ODEProblem(Mean_ODE, x0, Tspan)
        mean_and_var = solve(prob,solver,saveat=saveat)

        Mean=[]
        Var=[]
        for i in 1:length(mean_and_var)
            push!(Mean,diag(mean_and_var[i][1:no_of_species,1:no_of_species]))
            push!(Var,mean_and_var[i][no_of_species+1:end, 1:no_of_species])
        end

        return Mean, Var, mean_and_var.t

    end

    export LNAdecomp, LNA_Mean_Var


end




###################### Examples of using the two functions above ###################

using DifferentialEquations, ForwardDiff
params = [2.0, 1.0, 15.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
Tspan = (0.0, 10.0)

#x0[1:3] - 3 gene initial conditions, x0[3:6] initial conditions of noise
x0 = [0.0; 2.0; 1.0; 1.0; 1.0; 1.0]
volume=5.0
solver= DifferentialEquations.ImplicitEM()
dt=0.01

#stochiometry matrix
S=[1.0 -1.0 0.0 0.0 0.0 0.0 ; 0.0 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0]

#manual imput of rates of 3 three example
make_f = function(x,params)
    f=[params[1]/(1+params[7]*x[3]),
        params[4]*x[1], params[2]*params[8]*x[1]/(1+params[8]*x[1]), params[5]*x[2],
        params[3]*params[9]*x[1]*params[10]*x[2]/(1+params[9]*x[1])/(1+params[10]*x[2]),
        params[6]*x[3]]
    return f
end

#LNA decomp example
result,time=LNA.LNAdecomp(params,Tspan,x0, solver,dt,S,make_f, volume)
using Plots
meantrajectory= collect(Compat.Iterators.flatten(result))
sde= reshape(meantrajectory,3,length(result))
figure =plot(time,sde[1,:], c=:blue)
plot!(time,sde[2,:], c=:red)
plot!(time,sde[3,:], c=:green)


#LNA Mean Var example
saveat=0.01
solver= DifferentialEquations.ImplicitEuler()
x0 = [diagm([0.0, 2.0,1.0]) ; ones(3,3)]
Mean, Var, times = LNA.LNA_Mean_Var(params,Tspan,x0, solver,saveat,S,make_f,volume)


trajectory= collect(Compat.Iterators.flatten(Mean))
sde= reshape(trajectory,3,length(Mean))
var=[]
for elt in Var
    push!(var,diag(elt))
end
vartrajectory= collect(Compat.Iterators.flatten(var))
var= reshape(vartrajectory,3,length(Var))
plot!(times,sde[1,:],ribbon=1.96*sqrt.((var[1,:])/2.0), c=:blue, leg=false)
plot!(times,sde[2,:],ribbon=1.96*sqrt.((var[2,:])/2.0), c=:red)
plot!(times,sde[3,:],ribbon=1.96*sqrt.((var[3,:])/2.0), c=:green)
