import DiskArrayEngine as DAE
import DiskArrays: RegularChunks, IrregularChunks
using Test

@testset "Restart Files" begin
    # Create dummy Looprange object
    lr = DAE.ProductArray((RegularChunks(3, 0, 10), IrregularChunks(chunksizes=[1, 2, 3])))

    restartfile = tempname()

    restarter = DAE.create_restarter(restartfile, lr, :overwrite)

    DAE.add_entry(restarter, lr[1])
    DAE.add_entry(restarter, lr[3])
    DAE.add_entry(restarter, lr[5])

    lr_reloaded = DAE.orig_loopranges(restarter)

    @test lr_reloaded == lr
    @test DAE.finished_entries(restarter) == [(1:3, 1:1), (7:9, 1:1), (1:3, 2:3)]

    restarter = DAE.create_restarter(restartfile, lr, :continue)

    @test sort(restarter.remaining_loopranges) == [(1:3, 4:6), (4:6, 1:1), (4:6, 2:3), (4:6, 4:6), (7:9, 2:3), (7:9, 4:6), (10:10, 1:1), (10:10, 2:3), (10:10, 4:6)]

    # Test that integration into LocalRunner works
    input = reshape(1:30, 3, 10)
    output = zeros(3, 10)
    inar = DAE.InputArray(input)
    outspecs = DAE.create_outwindows(size(output), ismem=true)
    function f(input)
        if input == 28
            error("Test error")
        else
            return input
        end
    end
    func = DAE.create_userfunction(f, Int, is_mutating=false, allow_threads=true)
    op = DAE.GMDWop((inar,), (outspecs,), func)
    totsize = op.windowsize
    input_chunkspecs = DAE.get_chunkspec.(op.inars, (totsize,))
    output_chunkspecs = DAE.get_chunkspec.(op.outspecs, op.f.outtype)
    lr = DAE.ExecutionPlan(input_chunkspecs, output_chunkspecs, (1.0, 1.0), (2, 2), 0.1, DAE.ProductArray((RegularChunks(1, 0, 3), RegularChunks(4, 0, 10))))

    runner = DAE.LocalRunner(op, lr, (output,), restartfile=restartfile, restartmode=:overwrite)

    @test_throws "Test error" run(runner)
    @test output[1:24] == 1:24
    @test all(iszero, output[25:30])
    #Now fix the error in the function and run with restart
    function f2(input)
        #Make sure that restart only runs the parts that have not been run
        if input < 25
            error("Should not be hit")
        else
            return input
        end
    end
    func2 = DAE.create_userfunction(f2, Int, is_mutating=false, allow_threads=true)
    op2 = DAE.GMDWop((inar,), (outspecs,), func2)
    #Test that run fails if no restarter is applied
    runner2 = DAE.LocalRunner(op2, lr, (output,))
    @test_throws "Should not be hit" run(runner2)
    runner2 = DAE.LocalRunner(op2, lr, (output,), restartfile=restartfile, restartmode=:continue)
    run(runner2)
    @test output[1:30] == 1:30
    #Test that restart file is deleted in the end
    @test !isfile(restartfile)
end