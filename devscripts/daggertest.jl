using Dagger
using DiskArrayEngine
using DiskArrayEngine: generate_inbuffers, generate_outbuffers, read_range, extract_outbuffer, run_block, put_buffer, 
get_procgroups, DiskEngineScheduler, run_group

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
    inbuffers = Dagger.@shard generate_inbuffers(inars, loopranges)
    outspecs = op.outspecs
    f = op.f
    outbuffers = Dagger.@shard generate_outbuffers(outspecs,f, loopranges)
    DaggerRunner(op,loopranges,outars, threaded, inbuffers,outbuffers)
end


function DiskArrayEngine.run_loop(runner::DaggerRunner,loopranges = runner.loopranges;groupspecs=nothing)
    @debug "Groupspecs are ", groupspecs
    if groupspecs !== nothing && :output_chunk in groupspecs
        piddir = @spawn tempname()
    else
        piddir = nothing
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
    if (groupspecs !== nothing) && (:reducedim in groupspecs)
        @debug "Merging buffers"
        collections_merged = fetch(reduce(runner.outbuffers) do buf1, buf2
            map(buf1,buf2) do b1,b2
                res = merge_outbuffer_collection(b1,b2)
                @debug myid(), "merging output buffer collections of lengths ", length(b1), " ", length(b2), "to new length ", length(res)
                res
            end
        end)
        @debug "Writing merged buffers"
        for coll in collections_merged
            allkeys = collect(keys(coll))
            for k in allkeys
                @debug "Writing index $k"
                put_buffer((k,),runner.op.f.finalize, coll[k], coll, runner.outars, (piddir,))
            end
        end
    end
    true
end


function Base.run(runner::DaggerRunner)
    groups = get_procgroups(runner.op, runner.loopranges, runner.outars)
    sch = DiskEngineScheduler(groups, runner.loopranges, runner)
    run_group(sch)
end