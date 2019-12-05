# Frequently Asked Questions

#### Q: My computation terminates before sampling sufficient number of particles. How do I make it run longer?
**A:** All parameter inference functions accept an optional parameter `max_iter`, which specifies the number of iterations to run. See [ABC Reference](@ref abc_ref) for more details.

#### Q: When running the example on my machine, emulation provides no performance benefit or even takes longer than emulation. How do I speed up emulation?
**A:** The best way to reduce the time it takes to run the emulation is to decrease the `batch_size` parameter. Please refer to docs for [`EmulatedABCRejection`](@ref) end [`EmulatedABCSMC`](@ref). Machines with larger amount of RAM would be able to handle larger batches without performance degradation.

#### Q: How do I generate a plot that is similar to the one in the paper?
**A:** You will need to run ABC-SMC parameter estimation using both emulation and simulation for the same problem. Then, assuming you have [`Plots`](https://github.com/JuliaPlots/Plots.jl/) package installed, just run
```julia
julia> plot(emu_out, sim_out, true_params)
```
Use `pyplot()` [backend](https://docs.juliaplots.org/latest/backends/) for best results.

#### Q: How do I run the example notebooks locally?
**A:**
* Make sure that [Jupyter](https://jupyter.org/) is installed, along with [Julia](https://www.julialang.org/) and its [Jupyter Kernel](https://github.com/JuliaLang/IJulia.jl).
* Clone or download [GpABC.jl](https://github.com/tanhevg/GpABC.jl) to your machine.
* Run `jupyter` from the `examples` directory of the local copy of `GpABC.jl`:
```bash
$ cd GpABC.jl/examples
$ jupyter notebook
```
* The first line of the first cell, that contains something like
```julia
import Pkg; Pkg.activate("."); Pkg.resolve();
using ...
```
will download all the dependencies.

All notebooks were tested under Julia 1.2
