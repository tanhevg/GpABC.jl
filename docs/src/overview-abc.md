# [ABC Overview](@id abc-overview)

Approximate Bayesian Computation (ABC) is a collection of methods for likelihood free model parameter inference.

```@contents
Pages = ["overview-abc.md"]
```

## Simulation based Rejection ABC

The most basic variant of ABC is referred to as Rejection ABC. The user-defined inputs to this algorithm include:

- The prior distribution ``\pi``, defined over model parameter space ``\Theta``
- The model simulation function ``\textbf{f}``
- Reference data ``\mathcal{D}``
- Acceptance threshold ``\varepsilon``
- [Summary statistic](@ref summary_stats) ``\textbf{s}`` and distance function ``\textbf{d}``
- Desired size of the posterior sample and maximum number of simulations to perform

The pseudocode for simulation-based Rejection ABC in `GpABC` looks as follows:

- While the posterior sample is not full, and maximum number of simulations has not been reached:
  - Sample parameter vector (particle) ``\theta`` from ``\pi``
  - Simulate data ``x = \textbf{f}(\theta)``
  - Compute the distance between the summary statistic of the simulated data and that of the reference data `` y = \textbf{d}(\textbf{s}(x), \textbf{s}(\mathcal{D}))``
  - If ``y \leq \varepsilon``, then accept ``\theta`` in the posterior sample

This algorithm is implemented by Julia function [`SimulatedABCRejection`](@ref).

## Emulation based Rejection ABC

Some models are computationally expensive to simulate. Simulation based ABC for such models would take unreasonably long time to accept enough posterior particles.

To work around this issue, `GpABC` provides emulation based Rejection ABC. Rather than simulating the model for each sampled particle, this algorithm runs a small number of simulations in the beginning, and uses their results to train the emulator.

User-defined inputs for this algorithm are very similar to those for [Simulation based Rejection ABC](@ref):

- The prior distribution ``\pi``, defined over model parameter space ``\Theta``
- The model simulation function ``\textbf{f}``
- Reference data ``\mathcal{D}``
- Acceptance threshold ``\varepsilon``
- [Summary statistic](@ref summary_stats) ``\textbf{s}`` and distance function ``\textbf{d}``
- Number of design particles to sample: ``n``
- Batch size to use for regression: ``m``
- Desired size of the posterior sample and maximum number of regressions to perform

The pseudocode for emulation-based Rejection ABC in `GpABC` looks as follows:

- Sample ``n`` design particles from ``\pi``: ``\theta_1, ..., \theta_n``
- Simulate the model for the design particles: ``x_1, ..., x_n = \textbf{f}(\theta_1), ..., \textbf{f}(\theta_n)``
- Compute distances to the reference data: ``y_1, ..., y_n = \textbf{d}(\textbf{s}(x_1), \textbf{s}(\mathcal{D})), ..., \textbf{d}(\textbf{s}(x_n), \textbf{s}(\mathcal{D}))``
- Use ``\theta_1, ..., \theta_n`` and ``y_1, ..., y_n`` to train the emulator ``\textbf{gpr}``
  - *Advanced:* details of training procedure can be tweaked. See [`GpABC.train_emulator`](@ref).
- While the posterior sample is not full, and maximum number of regressions has not been reached:
  - Sample ``m`` particles from ``\pi``: ``\theta_1, ..., \theta_m``
  - Compute the approximate distances by running the emulator regression: ``y_1, ..., y_m = \textbf{gpr}(\theta_1), ..., \textbf{gpr}(\theta_m)``
  - For all ``j = 1 ... m``, if ``y_j \leq \varepsilon``, then accept ``\theta_j`` in the posterior sample
    - *Advanced:* details of the acceptance strategy can be tweaked. See [`GpABC.abc_select_emulated_particles`](@ref)

This algorithm is implemented by Julia function [`EmulatedABCRejection`](@ref).

## Simulation based ABC - SMC

This sophisticated version of ABC allows to specify a schedule of thresholds, as opposed to just a single value. A number of simulation based ABC iterations are then executed, one iteration per threshold. The posterior of the preceding iteration serves as a prior to the next one.

The user-defined inputs to this algorithm are similar to those of [Simulation based Rejection ABC](@ref):

- The prior distribution ``\pi``, defined over model parameter space ``\Theta``
- The model simulation function ``\textbf{f}``
- Reference data ``\mathcal{D}``
- A schedule of thresholds ``\varepsilon_1, ..., \varepsilon_T``
- [Summary statistic](@ref summary_stats) ``\textbf{s}`` and distance function ``\textbf{d}``
- Desired size of the posterior sample and maximum number of simulations to perform

The pseudocode for simulation-based ABC-SMC in `GpABC` looks as follows:

- For ``t`` in ``1 ... T``
  - While the posterior sample is not full, and maximum number of simulations has not been reached:
    - if ``t = 1``
      - Sample the particle ``\theta`` from ``\pi``
    - else
      - Sample the particle ``\theta`` from the posterior distribution of iteration ``t-1``
      - Perturb ``\theta`` using a perturbation kernel
    - Simulate data ``x = \textbf{f}(\theta)``
    - Compute the distance between the summary statistic of the simulated data and that of the reference data `` y = \textbf{d}(\textbf{s}(x), \textbf{s}(\mathcal{D}))``
    - If ``y \leq \varepsilon``, then accept ``\theta`` in the posterior sample

This algorithm is implemented by Julia function [`SimulatedABCSMC`](@ref).

## Emulation based ABC - SMC

Similarly to [Simulation based ABC - SMC](@ref), [Emulation based Rejection ABC](@ref) has an SMC counterpart. A threshold schedule must be supplied for this algorithm. A number of emulation based ABC iterations are then executed, one iteration per threshold. The posterior of the preceding iteration serves as a prior to the next one. Depending on user-defined settings, either the same emulator can be re-used for all iterations, or the emulator could be re-trained for each iteration.

The user-defined inputs to this algorithm are similar to those of [Emulation based Rejection ABC](@ref):

- The prior distribution ``\pi``, defined over model parameter space ``\Theta``
- The model simulation function ``\textbf{f}``
- Reference data ``\mathcal{D}``
- A schedule of thresholds ``\varepsilon_1, ..., \varepsilon_T``
- [Summary statistic](@ref summary_stats) ``\textbf{s}`` and distance function ``\textbf{d}``
- Number of design particles to sample: ``n``
- Batch size to use for regression: ``m``
- Desired size of the posterior sample and maximum number of regressions to perform

The pseudocode for emulation-based ABC-SMC in `GpABC` looks as follows:

- Sample ``n`` design particles from ``\pi``: ``\theta_1, ..., \theta_n``
- Simulate the model for the design particles: ``x_1, ..., x_n = \textbf{f}(\theta_1), ..., \textbf{f}(\theta_n)``
- Compute distances to the reference data: ``y_1, ..., y_n = \textbf{d}(\textbf{s}(x_1), \textbf{s}(\mathcal{D})), ..., \textbf{d}(\textbf{s}(x_n), \textbf{s}(\mathcal{D}))``
- Use ``\theta_1, ..., \theta_n`` and ``y_1, ..., y_n`` to train the emulator ``\textbf{gpr}``
  - *Advanced:* details of training procedure can be tweaked. See [`GpABC.train_emulator`](@ref).
- For ``t`` in ``1 ... T``
  - *Advanced*: optionally, if ``t > 1``, re-traing the emulator. See [`GpABC.abc_retrain_emulator`](@ref).
  - While the posterior sample is not full, and maximum number of regressions has not been reached:
    - Sample ``m`` particles from ``\pi``: ``\theta_1, ..., \theta_m``
    - Compute the approximate distances by running the emulator regression: ``y_1, ..., y_m = \textbf{gpr}(\theta_1), ..., \textbf{gpr}(\theta_m)``
    - For all ``j = 1 ... m``, if ``y_j \leq \varepsilon``, then accept ``\theta_j`` in the posterior sample
      - *Advanced:* details of the acceptance strategy can be tweaked. See [`GpABC.abc_select_emulated_particles`](@ref)

This algorithm is implemented by Julia function [`EmulatedABCSMC`](@ref).
