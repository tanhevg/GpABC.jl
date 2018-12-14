# Notation

In parts of this manual that deal with Gaussian Processes and kernels,
we denote the number of training points as $n$, and the number of
test points as $m$. The number of dimensions is denoted as $d$.

In the context of ABC, vectors in parameter space (``\theta``) are referred to as _particles_.
Particles that are used for training the emulator (`training_x`) are called _design points_.
To generate the distances for training the emulator (`training_y`), the model must be simulated for the design points.
