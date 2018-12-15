using RecipesBase

#=
!!!! IMPORTANT !!!!
Each code path of the @series macro must return the data for the series.
data = ... should be the last statement
=#
@recipe function abc_output_recipe(abco::ABCOutput;
      runs=nothing, params=nothing, population_colors=nothing)
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
                  subplot := (i - 1) * length(params) + j
                  if i == j
                        @series begin
                              seriestype := :histogram
                              xlabel --> "Parameter $(par1)"
                              ylabel := ""
                              yaxis := false
                              bins := 50
                              @debug repr(population_colors)
                              # if population_colors !== nothing
                              #       idx = max_color_idx % length(population_colors)
                              #       if idx == 0
                              #             idx = length(population_colors)
                              #       end
                              #       seriescolor := population_colors[idx]
                              # end
                              if isa(abco.population, Vector)
                                    data = abco.population[end][:, par1]
                                    # max_color_idx = 1
                              else
                                    data = abco.population[:, par1]
                                    # max_color_idx = length(abco.population)
                              end # if Vector
                        end  # @series
                  elseif j < i
                        seriestype := :scatter
                        markerstrokecolor --> false
                        if isa(abco.population, Vector)
                              for r in runs
                                    @series begin
                                          xlabel --> "Parameter $(par2)"
                                          ylabel --> "Parameter $(par1)"
                                          if population_colors !== nothing
                                                idx = r % length(population_colors)
                                                if idx == 0
                                                      idx = length(population_colors)
                                                end
                                                seriescolor := population_colors[idx]
                                          end
                                          pop = abco.population[r]
                                          data = (pop[:, par2], pop[:, par1])
                                    end # @series
                              end # for r
                        else
                              @series begin
                                    data = (abco.population[:, par2], abco.population[:, par1])
                              end # @series
                        end # if Vector
                  else
                        @series begin
                              ylabel := ""
                              xlabel := ""
                              grid := false
                              xaxis := false
                              yaxis := false
                              data = []
                        end # @series
                  end # if/elseif
            end # for j, par2
      end # for i, par1
end # @recipe

# Plot recipe for mdoel selection output
@recipe function modelselection_plotrecipe(::Type{ModelSelectionOutput}, mso::ModelSelectionOutput)
    seriestype := :line
    xlabel --> "Population"
    ylabel --> "Number of accepted particles"
    labels --> [string("Model ", m) for m in 1:mso.M]
    data = [[mso.n_accepted[i][j] for i in 1:size(mso.n_accepted,1)] for j in 1:mso.M]
end