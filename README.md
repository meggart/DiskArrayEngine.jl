# DiskArrayEngine.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/meggart/DiskArrayEngine.jl/blob/main/LICENSE)
<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://meggart.github.io/DiskArrayEngine.jl/stable) -->
<!-- [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://meggart.github.io/DiskArrayEngine.jl/dev) -->
[![CI](https://github.com/meggart/DiskArrayEngine.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/meggart/DiskArrayEngine.jl/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/meggart/DiskArrayEngine.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/meggart/DiskArrayEngine.jl)
<!-- [![Aqua.jl Quality Assurance](https://img.shields.io/badge/Aquajl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl) -->
[![Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/DiskArrayEngine&label=Downloads)](https://pkgs.genieframework.com?packages=DiskArrayEngine)

Tools for defining and running large computations on DiskArrays.

## Introduction

The package [`DiskArrays.jl`](https://github.com/meggart/DiskArrays.jl) implements Julia's AbstractArray interface for chunked (and possibly compressed) n-dimensional arrays that are stored on disk and operated on lazily. 
Although DiskArrays.jl provides basic implementations for e.g. broadcasting or reductions over dimensions it has clear limitations when it comes to parallel computations or when broadcasting over arrays from different sources with non-aligning chunks. With `DiskArrayEngine` intend to provide a general-purpose computing backend that scales to very large n-dimensional arrays (GBs, TBs or larger) typically stored in a DiskArrays.jl-supported format like NetCDF, Zarr, ArchGDAL, HDF5Utils etc with parallelism supported by Dagger.jl. 

## Scope of the package

Before starting to jump into this package it is worth checking if it is actually the right tool for your problem. Here is a quick check-list of things to consider and possible alternatives:

1. Your data is too large to fit into one machine's memory (otherwise just use normal Julia Arrays)
2. [`Mmap`](https://docs.julialang.org/en/v1/stdlib/Mmap/#Mmap.mmap) is not an option (e.g. because your data comes in a compressed format, or data is stored in the cloud or your queueing system sees unrealistic memory usage by mmap)
3. Your data is too large to fit into the memory of all your workers when distributed among them (otherwise try [DistributedArrays.jl](https://github.com/JuliaParallel/DistributedArrays.jl))
4. You want to process *all* or almost all of your data and not just a small subset. Otherwise just read the subset of interest into memory and do your processing based on this one

If you are still here 