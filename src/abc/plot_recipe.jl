using RecipesBase
using KernelDensity
using Random
using PlotUtils
using Colors

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
                              ylabel := "Accepted"
                              xlabel := "Parameter $(par1)"
                              bins --> 50
                              if isa(abco.population, Vector)
                                    data = abco.population[end][:, par1]
                              else
                                    data = abco.population[:, par1]
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

function scale_outer_intervals(intervals, scale=1.0)
    int_min = min([int[1] for int in intervals]...)
    int_max = max([int[2] for int in intervals]...)
    int_scaled_half = (int_max - int_min) * scale / 2.0
    int_mid = (int_min + int_max) / 2.0
    int_mid - int_scaled_half, int_mid + int_scaled_half
end

my_linspace(bounds::Tuple{Float64, Float64}, length::Int64) = range(bounds[1], stop=bounds[2], length=length)

@recipe function emulation_vs_simulation(emu_out::EmulatedABCSMCOutput, sim_out::SimulatedABCSMCOutput, true_params::AbstractVector{<:Real}=Float64[];
            pop_idx=0,
            pop_color = "#007731",
            simulation_color = "#08519c",
            emulation_color = "#800013",
            contour_levels=8)
      if pop_idx == 0
            pop_idx = min(length(sim_out.population), length(emu_out.population))
      end
      kernel_bandwidth_scale = 0.09
      bounds_scale = 1.2
      sim_count = 30
      params = 1:emu_out.n_params
      legend --> false
      layout := length(params) ^ 2
      for i in params
            for j in params
                  subplot := (i - 1) * length(params) + j
                  if j < i
                        x_data_emu = emu_out.population[pop_idx][:,j]
                        y_data_emu = emu_out.population[pop_idx][:,i]
                        x_data_sim = sim_out.population[pop_idx][:,j]
                        y_data_sim = sim_out.population[pop_idx][:,i]
                        sim_size = size(sim_out.population[pop_idx], 1)
                        if sim_size > sim_count
                            idx = randperm(sim_size)[1:sim_count]
                            x_data_sim = x_data_sim[idx]
                            y_data_sim = y_data_sim[idx]
                        end
                        x_extr_emu = extrema(x_data_emu)
                        y_extr_emu = extrema(y_data_emu)
                        x_extr_sim = extrema(x_data_sim)
                        y_extr_sim = extrema(y_data_sim)

                        bandwidth = (-kernel_bandwidth_scale * -(x_extr_emu...), -kernel_bandwidth_scale * -(y_extr_emu...))
                        kde_joint = kde((x_data_emu, y_data_emu), bandwidth=bandwidth)
                        contour_x = my_linspace(scale_outer_intervals([x_extr_emu], bounds_scale), 100)
                        contour_y = my_linspace(scale_outer_intervals([y_extr_emu], bounds_scale), 100)
                        contour_z = pdf(kde_joint, contour_x, contour_y)
                        @series begin
                              seriestype := :contour
                              color := ColorGradient([:white, emulation_color])
                              levels := contour_levels
                              fill := true
                              (contour_x, contour_y, contour_z)
                        end
                        @series begin
                              seriestype := :scatter
                              markersize := 4
                              markershape := :x
                              color := simulation_color
                              (x_data_sim, y_data_sim)
                        end
                        if length(true_params) > 0
                              seriestype := :path
                              linestyle := :dash
                              color := :black
                              markershape := :none
                              @series begin
                                    ([true_params[j], true_params[j], minimum(contour_x)],
                                    [minimum(contour_y), true_params[i], true_params[i]])
                              end
                        end
                  elseif j > i
                        pop_colors = range(parse(Colorant, "white"), parse(Colorant, pop_color), length=pop_idx+1)
                        seriestype := :scatter
                        markershape := :circle
                        markersize := 2
                        for k in 1:pop_idx
                              @series begin
                                    markercolor := pop_colors[k+1]
                                    markerstrokecolor := pop_colors[k+1]
                                    (emu_out.population[k][:,j], emu_out.population[k][:,i])
                              end
                        end
                        if length(true_params) > 0
                              seriestype := :path
                              linestyle := :dash
                              color := :black
                              markershape := :none
                              @series begin
                                    ([true_params[j], true_params[j], minimum(emu_out.population[1][:,j])],
                                    [minimum(emu_out.population[1][:,i]), true_params[i], true_params[i]])
                              end
                        end
                  else # i == j
                        seriestype := :path
                        markershape := :none
                        emu_data = emu_out.population[pop_idx][:,i]
                        sim_data = sim_out.population[pop_idx][:,i]
                        extr_emu = extrema(emu_data)
                        extr_sim = extrema(sim_data)
                        kde_emu = kde(emu_data, bandwidth=-kernel_bandwidth_scale * -(extr_emu...))
                        kde_sim = kde(sim_data, bandwidth=-kernel_bandwidth_scale * -(extr_sim...))
                        x_bounds = scale_outer_intervals([extr_emu, extr_sim], bounds_scale)
                        x_plot = my_linspace(x_bounds, 100)
                        y_emu_plot = pdf(kde_emu,x_plot)
                        y_sim_plot = pdf(kde_sim,x_plot)
                        @series begin
                              linestyle := :solid
                              color := simulation_color
                              (x_plot, y_sim_plot)
                        end
                        @series begin
                              linestyle := :solid
                              color := emulation_color
                              (x_plot, y_emu_plot)
                        end
                        if length(true_params) > 0
                              true_param = true_params[i]
                              max_pdf = max(pdf(kde_emu, true_param), pdf(kde_sim, true_param))
                              @series begin
                                    linestyle := :dash
                                    color := :black
                                    ([true_param, true_param], [0, max_pdf])
                              end
                        end
                  end # if/else/elseif i/j
            end # for j
      end # for i
end

# Plot recipe for mdoel selection output
@recipe function modelselection_plotrecipe(::Type{ModelSelectionOutput}, mso::ModelSelectionOutput)
    seriestype := :line
    xlabel --> "Population"
    ylabel --> "Number of accepted particles"
    labels --> [string("Model ", m) for m in 1:mso.M]
    data = [[mso.n_accepted[i][j] for i in 1:size(mso.n_accepted,1)] for j in 1:mso.M]
end
