@testset "Possible breaks" begin
    inwindows = 1:10
    outwindows = [1:3, 4:6, 7:10]
    pb = DAE.possible_breaks([inwindows], [outwindows])
    @test pb isa DAE.BlockMerge
    @test pb.possible_breaks == [3, 6, 10]

    inwindows = 1:4
    outwindows = [1, 2, 3, 4]
    pb = DAE.possible_breaks([inwindows], [outwindows])
    @test pb isa DAE.DirectMerge
end

@testset "Determining groups" begin
    strategies = [DAE.BlockMerge([2, 4, 6]), DAE.BlockMerge([1, 3, 8]), DAE.DirectMerge()]
    outconninwindows = (DAE.LoopWindows(DAE.ProductArray(([1:2, 3:4, 5:6], 1:8, 1:4)), Val((1, 2, 3))))
    inconnoutwindows = (DAE.LoopWindows(DAE.ProductArray((1:6, [1:1, 2:4, 4:8], 1:4)), Val((1, 2, 3))))
    inconngroups = DAE.get_groups(inconnoutwindows, strategies)
    outconngroups = DAE.get_groups(outconninwindows, strategies)
    @test inconngroups[1] == [1:2, 3:4, 5:6]
    @test inconngroups[2] == [1:1, 2:2, 3:3]
    @test outconngroups[1] == [1:1, 2:2, 3:3]
    @test outconngroups[2] == [1:1, 2:3, 4:8]

    newinwindows = DAE._blockwindows.([outconninwindows], (outconngroups,))
    newoutwindows = DAE._blockwindows.([inconnoutwindows], (inconngroups,))
    w = newinwindows[1].windows.members[1]
    wp = outconninwindows.windows.members[1]
    @test w[1] == DAE.WindowGroup(wp, 1:1)
    @test w[2] == DAE.WindowGroup(wp, 2:2)
    @test w[3] == DAE.WindowGroup(wp, 3:3)
    @test DAE.inner_values(w[1]) == [1:2]
    @test DAE.inner_range(w[1]) == 1:2
    @test DAE.inner_values(w[2]) == [3:4]
    @test DAE.inner_range(w[2]) == 3:4
    @test DAE.inner_values(w[3]) == [5:6]
    @test DAE.inner_range(w[3]) == 5:6

    w = newinwindows[1].windows.members[2]
    wp = outconninwindows.windows.members[2]
    @test w[1] == DAE.WindowGroup(wp, 1:1)
    @test w[2] == DAE.WindowGroup(wp, 2:3)
    @test w[3] == DAE.WindowGroup(wp, 4:8)

    w = newoutwindows[1].windows.members[1]
    wp = inconnoutwindows.windows.members[1]
    @test w[1] == DAE.WindowGroup(wp, 1:2)
    @test w[2] == DAE.WindowGroup(wp, 3:4)
    @test w[3] == DAE.WindowGroup(wp, 5:6)

    w = newoutwindows[1].windows.members[2]
    wp = inconnoutwindows.windows.members[2]
    @test w[1] == DAE.WindowGroup(wp, 1:1)
    @test w[2] == DAE.WindowGroup(wp, 2:2)
    @test w[3] == DAE.WindowGroup(wp, 3:3)
end

