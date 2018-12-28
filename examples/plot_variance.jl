using PyPlot

vars=[]
for i in 1:length(emu_out.population)
    m, v = gp_regression(emu_out.population[i], emu_out.emulators[i])
    push!(vars, v)
end

ion()
for i in 1:length(emu_out.threshold_schedule)
    epsilon = emu_out.threshold_schedule[i]
    figure()
    scatter3D(emu_out.population[i][:, 3], emu_out.population[i][:, 2], vars[i])
    title("Variance #$(i); ϵ=$(epsilon)")
    zlabel("Variance")
    xlabel("Parameter 3")
    ylabel("Parameter 2")
    figure(figsize=(13,4.8))
    subplot(121)
    scatter(emu_out.population[i][:, 3], emu_out.population[i][:, 2], marker=".")
    xlabel("Parameter 3")
    ylabel("Parameter 2")
    title("Accepted population #$(i); ϵ=$(epsilon)")
    subplot(122)
    scatter(emu_out.emulators[i].gp_training_x[:, 3], emu_out.emulators[i].gp_training_x[:, 2], marker="x")
    xlabel("Parameter 3")
    ylabel("Parameter 2")
    title("Emulator #$(i) training data")
end
