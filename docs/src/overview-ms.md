# [ABC Model Selection Overview](@id ms-overview)

The ABC SMC algorithm for model selection is available in full in the paper by Toni et al (see references for link). If using full model simulations the algorithm takes the following inputs:

* a prior over the models (the default in GpABC is a discrete uniform prior),
* a schedule of thresholds for each ABC run (the first is rejection ABC and the subsequent runs are ABC SMC),
* parameter priors for each candidate model,
* a maximum number of accepted particles per population, and
* a maximum number of iterations per population (default 1000).

As this is an ABC algorithm observed (reference) data, a distance metric and summary statistic must also be defined. As for other GpABC functions euclidean distance is the default distance metric.

The pseudocode of the model selection algorithm is

* Initialise thresholds ``\varepsilon_1,...,\varepsilon_T`` for ``T``x populations
* Initialise population index ``t=1``
* While ``t \leq T``
    * Initialise particle indicator ``i=1``
    * Initialise number of accepted particles for each of the ``M`` models ``A_1,...,A_M=0,...,0``
    * While ``\sum_m A_m < `` max no. of particles per population and $i<$ max no. of iterations per population
        * Sample model $m$ from the model prior
            * If ``t = 1``
                 * Perform rejection ABC for model $m$ using a single particle using threshold $\varepsilon_t$
                 * If particle is accepted
                     * ``A_m = A_m + 1``
            * Else
                * Perform ABC SMC for model $m$ with a single particle using threshold $\varepsilon_t$
                * If particle is accepted
                     * ``A_m = A_m + 1``
        * ``i = i + 1``
    * ``t = t + 1``
* Return number of accepted particles by each model at final population.


Note that for model selection the number of accepted particles applies across all the models, with the model accepting the maximum number of particles in the final population being the one that is best supported by the data.

### References

- Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. *Interface*, (July 2008), 187–202. [https://doi.org/10.1098/rsif.2008.0172](https://doi.org/10.1098/rsif.2008.0172)
