@testset "EngineArray mapreduce" begin
    using Zarr

    a = rand(Int, 10, 12)
    b = rand(Int, 10, 12)
    za = zzeros(Int, 10, 12, chunks=(4, 4))
    zb = zzeros(Int, 10, 12, chunks=(4, 4))
    za[:, :] = a
    zb[:, :] = b

    @test mapreduce(==, +, engine(za), engine(zb))[1] == mapreduce(==, +, a, b)
    @test mapreduce(==, +, engine(a), engine(b))[1] == mapreduce(==, +, a, b)
    @test_broken mapreduce(_ -> 1, + engine(za))[1] == 120

    for (m, n) in [(3, 4), (7, 7), (1, 5), (5, 1)]
        a = rand(Int, m, n)
        b = rand(Int, m, n)
        za = zzeros(Int, m, n, chunks=(2, 3))
        za[:, :] = a
        zb = zzeros(Int, m, n, chunks=(2, 3))
        zb[:, :] = b
        @test mapreduce(==, +, engine(za), engine(zb))[1] == mapreduce(==, +, a, b)
    end
end
