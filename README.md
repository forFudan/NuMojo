# Mojo-Arrays
A little project for vectorized nd arrays in native mojo
## Goals
* Vectorized, robustly statically typed, nd arrays with all of the standard matrix operations
* Eventually to be made compatible with mojo implementations or ABI of LAPACK and BLAS for linear algebra operations.
* Compatability with numpy arrays eventually for interactions with the rest of python. 
## Current
* DTypePointer-based data storage 2d arrays
* Basic guard rails for getting and setting
* add, subtract, mult, truediv, floordiv, and pow for both arrays of the same size, slices of arrays, and single values(but not commutatively yet for all)
* Arrange, transpose, shape, and meshgrid, and print basic implementations
* Getting and setting from slices
