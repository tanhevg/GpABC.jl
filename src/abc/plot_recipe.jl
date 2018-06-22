using RecipesBase

@recipe function abc_output_recipe(abco::ABCOutput, params=nothing)
      colors = (markercolor --> ["green" "blue" "red"])
      legend := false
      if params === nothing
            params = [i for i in 1:abco.n_params]
      end
      # layout := [1, 2]
      layout := length(params) ^ 2
      for (i, par1) in enumerate(params)
            for (j, par2) in enumerate(params)
                  @series begin
                        subplot := (i - 1) * length(params) + j
                        data = [0]
                        if i == j
                              seriestype := :histogram
                              bins := 50
                              data = abco.population[end][:, par1]
                        elseif j < i
                              seriestype := :scatter
                              x = Vector(length(colors))
                              y = Vector(length(colors))
                              for k in 1:length(colors)
                                    pop = abco.population[length(abco.population) - length(colors) + k]
                                    x[k] = pop[:, par1]
                                    y[k] = pop[:, par2]
                              end # for
                              data = (x, y)
                        else
                              foreground_color_subplot := false
                              grid := false
                        end # if/elseif
                        data
                  end # @series
            end # for j, par2
      end # for i, par1
end # @recipe
