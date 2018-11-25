* [Notation](@ref)
* [Basic Usage](@ref)
* [Training the GP](@ref)
* [Kernels](@ref)

**TODO markdown does not support include, so copy-paste content from /README.md once it is finalised**


## Notation

Throughout this manual, we denote the number of training points as $n$, and the number of
test points as $m$. The number of dimensions is denoted as $d$. For one-dimensional case, where each individual training and test point
is just a real number, both one-dimensional and two-dimensional arrays are accepted
as inputs. In [Basic Gaussian Process Regression Example](@ref example-1) `training_x` can either be a vector of size $n$, or an $n \times 1$
matrix. For a multidimensional case, where test and training points are elements of a
$d$-dimentional space, all inputs have to be row major, so `training_x` and `test_x` become
an $n \times d$ and an $m \times d$ matrices, respectively.
