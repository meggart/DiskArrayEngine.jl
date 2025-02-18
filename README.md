# DiskArrayEngine.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/meggart/DiskArrayEngine.jl/blob/main/LICENSE)
<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://meggart.github.io/DiskArrayEngine.jl/stable) -->
<!-- [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://meggart.github.io/DiskArrayEngine.jl/dev) -->
[![CI](https://github.com/meggart/DiskArrayEngine.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/meggart/DiskArrayEngine.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/meggart/DiskArrayEngine.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/meggart/DiskArrayEngine.jl)
<!-- [![Aqua.jl Quality Assurance](https://img.shields.io/badge/Aquajl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl) -->
[![Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/DiskArrayEngine&label=Downloads)](https://pkgs.genieframework.com?packages=DiskArrayEngine)

Tools for defining and running large computations on DiskArrays.

## Introduction

The package [`DiskArrays.jl`](https://github.com/meggart/DiskArrays.jl) implements Julia's AbstractArray interface for chunked (and possibly compressed) n-dimensional arrays that are stored on disk and operated on lazily. 
Although DiskArrays.jl provides basic implementations for e.g. broadcasting or reductions over dimensions, it has clear limitations when it comes to parallel computations or when broadcasting over arrays from different sources with non-aligning chunks. 

With `DiskArrayEngine`, we intend to provide a general-purpose computing backend that scales to very large n-dimensional arrays (GBs, TBs or larger), typically stored in a DiskArrays.jl-supported format like NetCDF, Zarr, ArchGDAL, HDF5, et cetera, with parallelism supported by `Dagger.jl`. 

## Scope of the package

Before starting to jump into this package, it is worth checking if it is actually the right tool for your problem.  Here is a quick check-list of things to consider and possible alternatives:

1. Your data is too large to fit into one machine's memory (otherwise just use normal Julia Arrays)
2. [`Mmap`](https://docs.julialang.org/en/v1/stdlib/Mmap/#Mmap.mmap) is not an option (e.g. because your data comes in a compressed format, or data is stored in the cloud or your queueing system sees unrealistic memory usage by mmap)
3. Your data is too large to fit into the memory of all your workers when distributed among them (otherwise try [DistributedArrays.jl](https://github.com/JuliaParallel/DistributedArrays.jl))
4. You want to process *all* or almost all of your data and not just a small subset. Otherwise just read the subset of interest into memory and do your processing based on this one

If you are still here, you should also note that this package is not intended to be used by end-users directly, but the plan is to wrap functionality from this package 
in other packages. In particular, these are YAXArrays.jl, DimensionalData.jl,or PyramidSchemes.jl, that provide more user-friendly interfaces. 

## Status of the package

This package is still under active development and should be considered experimental. Expect things to break and to already be broken. In particular, extensive documentation and tests are still missing. However, some core functionality of the package is already used by e.g. [PyramidScheme.jl](https://github.com/JuliaDataCubes/PyramidScheme.jl) which is why we decided to already register this package while still under active development. 

## Basic package usage

To be done, describe the generalized moving window concept, how to define user functions, lazy interface and which runner options exist.

## EngineArrays

The simplest way to use some the machinery in DiskArrayEngine is to wrap any existing DiskArray into an EngineArray by calling `engine(mydiskarray)`. Afterwards, many operations like mapslices, mapreduce, broadcast and simple statistics like mean, median, max/min etc will be dispatch using DiskArrayEngine instead of the simple 
DiskArrays.jl implementation and might give significant speedups. However, currently we still default to using the `LocalRunner`, which will only use a single process, we will experiment with defaulting to `DaggerRunner` as soon as multiple processes are available. 
