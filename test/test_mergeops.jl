@testset "BlockMerge Run" begin
    inwindows1 = DAE.MovingWindow(1, 5, 5, 4)
    outwindows1 = 1:4
    inwindows2 = DAE.MovingWindow(1, 2, 2, 2)
    outwindows2 = 1:2

    inar = DAE.InputArray(1:20, windows=(inwindows1,))
    outspecs = (DAE.create_outwindows(4, windows=(outwindows1,)),)
    f = create_userfunction(sum, Float64)
    op1 = DAE.GMDWop((inar,), outspecs, f)
    r = results_as_diskarrays(op1)[1]

    inar2 = DAE.InputArray(r, windows=(inwindows2,))
    outspecs2 = (DAE.create_outwindows(4, windows=(outwindows2,)),)
    f2 = create_userfunction(sum, Float64)
    op2 = DAE.GMDWop((inar2,), outspecs2, f2)
    r2 = results_as_diskarrays(op2)[1]

    g = DAE.result_to_graph(r2)

    @test length(g.nodes) == 3
    @test g.nodes[1] == DAE.MwopOutNode(false, nothing, (2,), Float64)
    @test g.nodes[2] == DAE.MwopOutNode(false, nothing, (4,), Float64)
    @test g.nodes[3] == 1:20

    @test length(g.connections) == 2
    conn1, conn2 = g.connections
    @test conn1.inputids == [3]
    @test conn1.outputids == [2]
    @test conn2.inputids == [2]
    @test conn2.outputids == [1]

    nodemergestrategies = DAE.collect_strategies(g)

    @test only(nodemergestrategies[2]) isa DAE.BlockMerge
    @test nodemergestrategies[1] == [nothing]
    @test nodemergestrategies[3] == [nothing]

    dimmap = DAE.create_loopdimmap(conn1, conn2, 2)
    @test dimmap isa DAE.DimMap
    @test dimmap.d == Dict(1 => 1)

    newop = DAE.merge_operations(DAE.BlockMerge, conn1, conn2, 2, dimmap)
    @test newop isa DAE.UserOp
    @test newop.f isa DAE.BlockFunctionChain
    @test newop.f.funcs[1] === f.f
    @test newop.f.funcs[2] === f2.f
    @test newop.f.args == [((1,), (1,)), ((2,), (2,))]
    @test newop.f.transfers == [1 => [2]]

    newconn, newnodes = DAE.merged_connection(DAE.BlockMerge, g, conn1, conn2, 2, newop, nodemergestrategies, dimmap)

    @test newconn isa DAE.MwopConnection
    @test newconn.f === newop
    @test newconn.inputids == [3, 4]
    @test newconn.outputids == [2, 1]

    win1 = newconn.inwindows[1].windows.members[1]
    @test win1 isa DAE.Window
    @test eltype(win1) <: DAE.WindowGroup
    @test length(win1) == 2
    @test win1[1].g == 1:2
    @test win1[1].parent == [1:5, 6:10, 11:15, 16:20]
    @test win1[2].g == 3:4
    @test win1[2].parent == [1:5, 6:10, 11:15, 16:20]
    @test DAE.avg_step(win1) == 10
    @test DAE.max_size(win1) == 10


    wout1 = newconn.outwindows[1].windows.members[1]
    @test wout1 isa DAE.Window
    @test eltype(wout1) <: DAE.WindowGroup
    @test length(wout1) == 2
    @test wout1[1].g == 1:2
    @test wout1[1].parent == 1:4
    @test wout1[2].g == 3:4
    @test wout1[2].parent == 1:4
    @test DAE.avg_step(wout1) == 2
    @test DAE.max_size(wout1) == 2

    win2 = newconn.inwindows[2].windows.members[1]
    @test win2 isa DAE.Window
    @test eltype(win2) <: DAE.WindowGroup
    @test length(win2) == 2
    @test win2[1].g == 1:1
    @test win2[1].parent == [1:2, 3:4]
    @test win2[2].g == 2:2
    @test win2[2].parent == [1:2, 3:4]
    @test DAE.avg_step(win2) == 2
    @test DAE.max_size(win2) == 2

    wout2 = newconn.outwindows[2].windows.members[1]
    @test wout2 isa DAE.Window
    @test eltype(wout2) <: DAE.WindowGroup
    @test length(wout2) == 2
    @test wout2[1].g == 1:1
    @test wout2[1].parent == 1:2
    @test wout2[2].g == 2:2
    @test wout2[2].parent == 1:2
    @test DAE.avg_step(wout2) == 1
    @test DAE.max_size(wout2) == 1

    @test length(newnodes) == 1
    @test newnodes[1] == DAE.EmptyInput{Float64,1}((4,))

    append!(g.nodes, newnodes)

    deleteat!(g.connections, [1, 2])
    push!(g.connections, newconn)

    newop = DAE.gmwop_from_reducedgraph(g)

    inar = newop.inars[1]
    cspec = DAE.get_chunkspec(inar, (2,))
    @test cspec.app_cs == (2,)
    @test cspec.windowfac == (10,)

    lr = DAE.custom_loopranges(newop, (1,))

    @test DAE.windowbuffersize(lr.lr.members[1], inar.lw.windows.members[1]) == 10

    runner = DAE.LocalRunner(newop, lr, (nothing, zeros(2)))

    for inow in lr.lr
        op = newop
        inbuffers_pure = runner.inbuffers_pure
        outbuffers = runner.outbuffers
        inbuffers_wrapped = DAE.read_range.((inow,), op.inars, inbuffers_pure)
        outbuffers_now = DAE.extract_outbuffer.((inow,), op.outspecs, op.f.init, op.f.buftype, outbuffers)
        DAE.run_block(op, inow, inbuffers_wrapped, outbuffers_now, false)
        DAE.put_buffer.((inow,), outbuffers_now, runner.outars, nothing)
    end

    @test runner.outars == (nothing, [55.0, 155.0])
end