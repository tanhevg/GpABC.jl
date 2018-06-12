**TODO markdown does not support include, so copy-paste content from /README.md once it is finalised**

## Basic Usage

The package is built around a type [`GPModel`](@ref), which encapsulates all the information
required for training the Gaussian Process and performing the regression. In the simplest
scenario the user would instantiate this type with some training data and labels, provide
the hyperparameters and run the regression. By default, [`SquaredExponentialIsoKernel`](@ref) will be used. This scenario is illustrated by [Basic Gaussian Process Regression Example](@ref example-1).

## Notation

Throughout this manual, we denote the number of training points as $n$, and the number of
test points as $m$. The number of dimensions is denoted as $d$. For one-dimensional case, where each individual training and test point
is just a real number, both one-dimensional and two-dimensional arrays are accepted
as inputs. In [Basic Gaussian Process Regression Example](@ref example-1) `training_x` can either be a vector of size $n$, or an $n \times 1$
matrix. For a multidimensional case, where test and training points are elements of a
$d$-dimentional space, all inputs have to be row major, so `training_x` and `test_x` become
an $n \times d$ and an $m \times d$ matrices, respectively.

[Examples](@ref)
