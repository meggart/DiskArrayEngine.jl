using Dagger: Dagger, shard

struct DaggerRunner{OP,LR,OA,IB,OB}
    op::OP
    loopranges::LR
    outars::OA
    threaded::Bool
    inbuffers_pure::IB
    outbuffers::OB
end
function DaggerRunner(op,loopranges,outars;threaded=true)
    inars = op.inars
    inbuffers = Dagger.shard(per_thread=true) do
        generate_inbuffers(inars, loopranges)
    end
    outspecs = op.outspecs
    f = op.f
    outbuffers = Dagger.shard(per_thread=true) do 
        generate_outbuffers(outspecs,f, loopranges)
    end
    DaggerRunner(op,loopranges,outars, threaded, inbuffers,outbuffers)
end


function run_loop(runner::DaggerRunner,loopranges = runner.loopranges;groupspecs=nothing)
    @debug "Groupspecs are ", groupspecs
    piddir = if groupspecs !== nothing && :output_chunk in groupspecs
        tempname()
    else
        nothing
    end
    op = runner.op
    fetch.(broadcast(loopranges) do inow
        Dagger.spawn(runner.inbuffers_pure,runner.outbuffers,inow,piddir,runner.outars,loopranges) do inbuffers_pure, outbuffers, inow, piddir, outars,loopranges
            @debug myid(), " Starting block ", inow

            inbuffers_wrapped = read_range.((inow,),op.inars,inbuffers_pure);
            outbuffers_now = extract_outbuffer.((inow,),(loopranges,),op.outspecs,op.f.init,op.f.buftype,outbuffers)
            run_block(op,inow,inbuffers_wrapped,outbuffers_now,true)
            @debug myid(), "Finished running block ", inow

            put_buffer.((inow,),op.f.finalize, outbuffers_now, outbuffers, outars, (piddir,))
            true
        end
    end)
    if (groupspecs !== nothing) && any(i->in(:reducedim,i.reasons),groupspecs)
        @debug "Merging buffers"
        red = op.f.red
        collections_merged = fetch(merge_all_outbuffers(runner.outbuffers,op.f.red))
        @debug "Writing merged buffers"
        flush_all_outbuffers(collections_merged,op.f.finalize,runner.outars,piddir)
    end
    GC.gc()
    true
end

function Base.run(runner::DaggerRunner)
    groups = get_procgroups(runner.op, runner.loopranges, runner.outars)
    sch = DiskEngineScheduler(groups, runner.loopranges, runner)
    run_group(sch)
end

function schedule(sch::DiskEngineScheduler,::DaggerRunner,loopdims,loopsub,groupspecs)
    fetch.(map(loopsub) do i
        lrsub = subset_loopranges(sch.loopranges,loopdims,i.I)
        schsub = DiskEngineScheduler(sch.groups,lrsub,sch.runner)
        Dagger.spawn(schsub,groupspecs) do sched, gs
            run_group(sched;gs)
        end
    end)
end