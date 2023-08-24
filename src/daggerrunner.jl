using Dagger: Dagger, shard

struct DaggerRunner{OP,LR,OA,IB,OB}
    op::OP
    loopranges::LR
    outars::OA
    threaded::Bool
    workerthreads::Bool
    inbuffers_pure::IB
    outbuffers::OB
end
function DaggerRunner(op,loopranges,outars;workerthreads=false,threaded=true)
    inars = op.inars
    inbuffers = Dagger.shard(per_thread=workerthreads) do
        generate_inbuffers(inars, loopranges)
    end
    outspecs = op.outspecs
    f = op.f
    outbuffers = Dagger.shard(per_thread=workerthreads) do 
        generate_outbuffers(outspecs,f, loopranges)
    end
    DaggerRunner(op,loopranges,outars, threaded, workerthreads, inbuffers,outbuffers)
end

buffer_mergefunc(red,::Type{<:Union{Dagger.Chunk,Dagger.Thunk, Dagger.EagerThunk}}) = (buf1,buf2) -> begin
@debug "Creating Dagger merge function"
  Dagger.spawn(buf1,buf2) do x,y
    merge_outbuffer_collection.(x,y,(red,))
  end
end

function run_loop(runner::DaggerRunner,loopranges = runner.loopranges;groupspecs=nothing)
    @debug "Groupspecs are ", groupspecs
    piddir = if groupspecs !== nothing && :output_chunk in groupspecs
        tempname()
    else
        nothing
    end
    op = runner.op
    r = broadcast(loopranges) do inow
        Dagger.spawn(runner.inbuffers_pure,runner.outbuffers,inow,piddir,runner.outars,loopranges) do inbuffers_pure, outbuffers, inow, piddir, outars,loopranges
            @debug myid(), " Starting block ", inow
            inbuffers_wrapped = read_range.((inow,),op.inars,inbuffers_pure);
            outbuffers_now = extract_outbuffer.((inow,),(loopranges,),op.outspecs,op.f.init,op.f.buftype,outbuffers)
            run_block(op,inow,inbuffers_wrapped,outbuffers_now,runner.threaded)
            @debug myid(), "Finished running block ", inow

            put_buffer.((inow,),op.f.finalize, outbuffers_now, outbuffers, outars, (piddir,))
            true
        end
    end
    @debug myid(), " Finished spawning jobs"
    all(fetch.(r)) || error("Error during chunk processing")
    @debug myid(), " Fetched everything"
    if false && (groupspecs !== nothing) && any(i->in(:reducedim,i.reasons),groupspecs)
        @debug "Merging buffers"
        procs = unique(Dagger.processor.(fetch.(r,raw=true)))
        @debug "Affected processors are $procs"
        red = op.f.red
        buffers_used = collect(v for (k,v) in runner.outbuffers.chunks if any(p->matches_proc(k,p),procs))
        buffer_copies = fetch.(map(buffers_used) do buf
            Dagger.spawn(buf) do b
                @debug "Copying buffers"
                c = deepcopy(b)
                @debug "Emptying buffers"
                foreach(b) do oa
                    empty!(oa.buffers)
                end
                c
            end
        end)
        @debug "Merging buffers from $(length(buffers_used)) workers."
        collections_merged = merge_all_outbuffers(buffer_copies,op.f.red)
        @debug "Writing merged buffers"
        unflushed_buffers = flush_all_outbuffers(collections_merged,op.f.finalize,runner.outars,piddir)
        @debug "Putting back flushed buffers"
        r = Dagger.spawn(unflushed_buffers,runner.outbuffers,red) do rembuf,outbuf, red
            foreach(rembuf,outbuf) do r,o
                if !isempty(r.buffers)
                    @debug "Putting back unflushed data"
                    newagg = merge_outbuffer_collection(o,r,red)
                    empty!(o)
                    for k in keys(newagg)
                        o[k] = newagg
                    end
                end
            end
        end
        fetch(r)
    end
    GC.gc()
    true
end

matches_proc(k::Dagger.ThreadProc,c::Dagger.ThreadProc) = k==c
matches_proc(k::Dagger.OSProc,c::Dagger.ThreadProc) = c.tid != 1 ? error("Processing was not on tid 1") : c.owner == k.pid

function Base.run(runner::DaggerRunner)
    @debug "Starting to run"
    groups = get_procgroups(runner.op, runner.loopranges, runner.outars)
    sch = DiskEngineScheduler(groups, runner.loopranges, runner)
    opts = runner.workerthreads ? (;) : (;scope = Dagger.scope(thread=1))
    Dagger.with_options(;opts...) do
        @debug "Calling first run_group"
        run_group(sch)
    end
end

function schedule(sch::DiskEngineScheduler,::DaggerRunner,loopdims,loopsub,groupspecs)
    @debug "Starting to schedule: "
    fetch.(map(loopsub) do i
        lrsub = subset_loopranges(sch.loopranges,loopdims,i.I)
        @debug "New split loopranges are: ", lrsub.members
        schsub = DiskEngineScheduler(sch.groups,lrsub,sch.runner)
        @debug "Spawning"
        Dagger.spawn(schsub,groupspecs) do sched, gs
            run_group(sched;groupspecs = gs)
        end
    end)
end