using DiskArrayEngine
using DiskArrays: AbstractDiskArray, DiskArrayEngineBackend, withbackend
using DiskArrays.TestTypes: AccessCountDiskArray
using Statistics
using Test

import DiskArrayEngine as DAE

# ── Helpers ─────────────────────────────────────────────────────────────────

function make_arrays(data; chunksize=size(data))
    mat = data
    # Select the backend per array, so the tests do not depend on the backend preference
    da = withbackend(AccessCountDiskArray(data; chunksize=chunksize), DiskArrayEngineBackend())
    return (materialized=mat, disk=da)
end

# ── Scalar reductions ───────────────────────────────────────────────────────

function test_scalar_reductions(data; chunksize=ntuple(i -> max(1, size(data)[i] ÷ 2), ndims(data)))
    @testset "DiskArrayEngine: scalar reductions" begin
        mat, da = make_arrays(data; chunksize)

        @testset "sum" begin
            @test sum(da) ≈ sum(mat)
            @test sum(identity, da) ≈ sum(mat)
            @test sum(x -> 2x, da) ≈ sum(x -> 2x, mat)
        end

        @testset "prod" begin
            @test prod(da) ≈ prod(mat)
            @test prod(identity, da) ≈ prod(mat)
            @test prod(x -> 2x, da) ≈ prod(x -> 2x, mat)
        end

        @testset "minimum / maximum" begin
            @test minimum(da) ≈ minimum(mat)
            @test minimum(identity, da) ≈ minimum(mat)
            @test minimum(x -> abs(x), da) ≈ minimum(x -> abs(x), mat)
            @test maximum(da) ≈ maximum(mat)
            @test maximum(identity, da) ≈ maximum(mat)
            @test maximum(x -> abs(x), da) ≈ maximum(x -> abs(x), mat)
            # The OnlineStats path, `:auto` mostly picks the direct one for these small arrays
            @test minimum(da; strategy=:reduce) ≈ minimum(mat)
            @test maximum(da; strategy=:reduce) ≈ maximum(mat)
        end

        @testset "extrema" begin
            @test extrema(da) == extrema(mat)
            @test extrema(identity, da) == extrema(mat)
            @test extrema(x -> abs(x), da) == extrema(x -> abs(x), mat)
        end

        @testset "count" begin
            @test count(x -> x > 0, da) == count(x -> x > 0, mat)
        end

        @testset "mean" begin
            @test mean(da) ≈ mean(mat)
            @test mean(identity, da) ≈ mean(mat)
            @test mean(x -> 2x, da) ≈ mean(x -> 2x, mat)
        end

        @testset "median" begin
            @test median(da) ≈ median(mat)
        end
    end
end

# ── Lazy result verification ────────────────────────────────────────────────

function test_lazy_results(data; chunksize=ntuple(i -> max(1, size(data)[i] ÷ 2), ndims(data)))
    @testset "DiskArrayEngine: lazy results" begin
        da = make_arrays(data; chunksize).disk

        # sum/mean with dims≠: should return lazy GMWOPResult (DAE's impl
        # calls aggregate_diskarray which returns a lazy array when dims is set).
        # The DefaultBackend fallback would return a materialized Matrix, so
        # checking the type proves the DAE impl path is exercised.
        s = sum(da, dims=1)
        @test s isa DAE.GMWOPResult
        @test Array(s) ≈ sum(data, dims=1)

        m = mean(da, dims=1)
        @test m isa DAE.GMWOPResult
        @test Array(m) ≈ mean(data, dims=1)
    end
end

# ── mapreduce ───────────────────────────────────────────────────────────────

function test_mapreduce(data; chunksize=ntuple(i -> max(1, size(data)[i] ÷ 2), ndims(data)))
    @testset "DiskArrayEngine: mapreduce" begin
        mat, da = make_arrays(data; chunksize)

        @testset "mapreduce (no dims, no init)" begin
            @test mapreduce(x -> 2x, +, da) ≈ mapreduce(x -> 2x, +, mat)
            # `op` must be associative, chunks are reduced in an arbitrary order
            @test mapreduce(abs, max, da) ≈ mapreduce(abs, max, mat)
        end

        @testset "mapreduce (dims=)" begin
            @test Array(mapreduce(x -> 2x, +, da; dims=1)) ≈ mapreduce(x -> 2x, +, mat; dims=1)
            @test Array(mapreduce(x -> 2x, +, da; dims=(1, 2))) ≈ mapreduce(x -> 2x, +, mat; dims=(1, 2))
        end

        @testset "mapreduce (init)" begin
            @test mapreduce(identity, +, da; init=0) ≈ mapreduce(identity, +, mat; init=0)
            @test mapreduce(identity, *, da; init=1) ≈ mapreduce(identity, *, mat; init=1)
            # An Int `init` with float data and `dims` is an InexactError in Base as well
            @test Array(mapreduce(identity, +, da; dims=1, init=0.0)) ≈ mapreduce(identity, +, mat; dims=1, init=0.0)
        end

        @testset "mapreducedim!" begin
            # Reduce over the last and over the first dimension
            redsize(d) = ntuple(i -> i == d ? 1 : size(mat, i), ndims(mat))
            R = zeros(redsize(ndims(mat)))
            Base.mapreducedim!(x -> 2x, +, R, da)
            @test R ≈ Base.mapreducedim!(x -> 2x, +, zero(R), mat)
            R2 = zeros(redsize(1))
            Base.mapreducedim!(x -> x^2, +, R2, da)
            @test R2 ≈ Base.mapreducedim!(x -> x^2, +, zero(R2), mat)
        end

        @testset "mapfoldl (no init)" begin
            @test mapfoldl(x -> 2x, +, da) ≈ mapfoldl(x -> 2x, +, mat)
        end

        @testset "mapfoldl (with init)" begin
            @test mapfoldl(identity, +, da; init=0) ≈ mapfoldl(identity, +, mat; init=0)
            @test mapfoldl(identity, *, da; init=1) ≈ mapfoldl(identity, *, mat; init=1)
        end
    end
end

# ── Run tests over multiple array shapes ────────────────────────────────────

@testset "DiskArrayEngine: scalar reductions" begin
    for data in (randn(5, 4, 2), randn(10), randn(10, 20), randn(3, 3, 3))
        test_scalar_reductions(data)
    end
end

@testset "DiskArrayEngine: lazy results" begin
    for data in (randn(5, 4, 2), randn(10), randn(10, 20), randn(3, 3, 3))
        test_lazy_results(data)
    end
end

@testset "DiskArrayEngine: mapreduce" begin
    for data in (randn(5, 4, 2), randn(10), randn(10, 20), randn(3, 3, 3))
        test_mapreduce(data)
    end
end
