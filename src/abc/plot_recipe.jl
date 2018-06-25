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
                              bins := 50
                              if isa(abco.population, Vector)
                                    data = abco.population[end][:, par1]
                              else
                                    data = abco.population[:, par1]
                              end # if Vector
                        elseif j < i
                              seriestype := :scatter
                              if isa(abco.population, Vector)
                                    x = Vector(length(runs))
                                    y = Vector(length(runs))
                                    for (r, run) in enumerate(runs)
                                          pop = abco.population[run]
                                          x[r] = pop[:, par1]
                                          y[r] = pop[:, par2]
                                    end # for r
                                    data = (x, y)
                              else
                                    data = (abco.population[:, par1], abco.population[:, par2])
                              end # if Vector
                        else
                              data = [0]
                              foreground_color_subplot := false
                              grid := false
                        end # if/elseif
                        data
                  end # @series
            end # for j, par2
      end # for i, par1
end # @recipe
