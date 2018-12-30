# [Stochastic Inference (LNA) Overview](@id lna-overview)

The LNA approximates the Chemical Master Equation (CME) by decomposing the stochastic process into two ordinary differential equations (ODEs); one describing the evolution of the mean of the trajectories and the other describing the evolution of the covaraince of the trajectories.

In other words the LNA approximates the stochastic process by looking at the mean and the covariance of the trajectories $\textbf{x}(t)$, whose evolution is described by a system of ODEs which can be seen below:

```math
\begin{align*}
\frac{d\phi(t)}{dt}&=S\cdot \mathbf{a}(\phi(t)) \label{mean} \\
\frac{d\Sigma(t)}{dt}&=S\cdot J \cdot \Sigma(t) + \Sigma(t) \cdot (J\cdot S)^T+
\Omega^{-1/2} S\cdot \mathrm{diag} \{\mathbf{a}(\phi(t))\} \cdot S^T \label{covar}
\end{align*}
```

Here $S$ is the stoichometry matrix of the system, $\textbf{a}$ is the reaction propensity vector.

The $J(t)_{jk}=\partial a_j/\partial \phi_k$ is the Jacobian of the $j^{th}$ reaction with respect to the $k^{th}$ variable.

These can be solved by numerical methods to describe how $phi(t)$ (the mean) and $\Sigma(t)$ (the covariance) evolve with time.

### References

- Komorowski, M., Finkenstädt, B., Harper, C.V., and Rand, D.A. (2009). Bayesian inference of biochemical kinetic parameters using the
linear noise approximation. *BMC Bioinformatics*, 10:343.

- Schnoerr, D., Sanguinetti, G., and Grima, R. (2017). Approximation and inference methods for stochastic biochemical kinetics—a tutorial review. *Journal of Physics A: Mathematical and Theoretical*, 50(9), 093001.
