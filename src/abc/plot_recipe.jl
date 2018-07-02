using RecipesBase

@recipe function abc_output_recipe(abco::ABCOutput;
            runs=nothing, params=nothing)
      if params === nothing
            params = [i for i in 1:abco.n_params]
      end
      if runs === nothing
            runs = [i for i in 1:length(abco.population)]
      elseif isa(runs, Int)
            runs = [i for i in max(length(abco.population) - runs + 1, 1):length(abco.population)]
      end
      legend --> false
      layout := length(params) ^ 2
      for (i, par1) in enumerate(params)
            for (j, par2) in enumerate(params)
                  @series begin
                        subplot := (i - 1) * length(params) + j
                        if i == j
                              seriestype := :histogram
                              xlabel --> "Parameter $(par1)"
                              yaxis := false
                              bins := 50
                              if isa(abco.population, Vector)
                                    data = abco.population[end][:, par1]
                              else
                                    data = abco.population[:, par1]
                              end # if Vector
                        elseif j < i
                              seriestype := :scatter
                              markerstrokecolor --> false
                              xlabel --> "Parameter $(par2)"
                              ylabel --> "Parameter $(par1)"
                              if isa(abco.population, Vector)
                                    x = Vector()
                                    y = Vector()
                                    for r in runs
                                          pop = abco.population[r]
                                          append!(x, [pop[:, par2]])
                                          append!(y, [pop[:, par1]])
                                    end # for r
                                    data = (x, y)
                              else
                                    data = (abco.population[:, par2], abco.population[:, par1])
                              end # if Vector
                        else
                              data = [0]
                              grid := false
                              xaxis := false
                              yaxis := false
                        end # if/elseif
                        data
                  end # @series
            end # for j, par2
      end # for i, par1
end # @recipe
