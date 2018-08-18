using PyPlot

function plot_distances_training_accepted(emu_out::ABCSMCOutput)
    n_iter = length(emu_out.threshold_schedule)
    for smc_iteration in 1:n_iter
        for parameter_idx in 1:emu_out.n_params
            subplot_idx = (smc_iteration - 1) * emu_out.n_params + parameter_idx
            subplot(n_iter, emu_out.n_params, subplot_idx)
            title("SMC iteration $(smc_iteration)")
            scatter(emu_out.emulators[smc_iteration].gp_training_x[:, parameter_idx],
                emu_out.emulators[smc_iteration].gp_training_y, marker=".",
                label="Emulation - training")
            scatter(emu_out.population[smc_iteration][:, parameter_idx],
                emu_out.distances[smc_iteration], marker=".",
                label="Emulation - accepted")
            xlabel("Theta_$(parameter_idx)")
            ylabel("SMC iteration $(smc_iteration) - L2 distance")
            legend()
        end
    end
end

ion()
f = figure()
ioff()
plot_distances_training_accepted(emu_out)
subplots_adjust(
left    =  0.08,
bottom  =  0.06,
right   =  0.96,
top     =  0.97,
wspace  =  0.26,
hspace  =  0.3
)
show(f)
