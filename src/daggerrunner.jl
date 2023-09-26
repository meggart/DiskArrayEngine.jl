using Dagger: Dagger, shard

struct DaggerRunner
    op
    loopranges
    outars
    threaded::Bool
    workerthreads::Bool
    inbuffers_pure
end
make_outbuffer_shard(op,runnerloopranges,workerthreads) = Dagger.shard(per_thread=workerthreads) do 
    generate_outbuffers(op.outspecs,op.f,runnerloopranges)
end
function DaggerRunner(op,exec_plan,outars=create_outars(op,exec_plan);workerthreads=false,threaded=true)
    inars = op.inars
    loopranges = plan_to_loopranges(exec_plan)
    inbuffers = Dagger.shard(per_thread=workerthreads) do
        generate_inbuffers(inars, loopranges)
    end
    try
        generate_outbuffers(op.outspecs,op.f, loopranges)
    catch e
        rethrow(e)
    end
    DaggerRunner(op,loopranges,outars, threaded, workerthreads, inbuffers)
end

buffer_mergefunc(red,::Type{<:Union{Dagger.Chunk,Dagger.Thunk, Dagger.EagerThunk}}) = (buf1,buf2) -> begin
    @debug "Creating Dagger merge function"
    @debug "Fetching x"
    fx = fetch(x)
    @debug "Fetching y"
    fy = fetch(y)
    @debug "Calling function"
    merge_outbuffer_collection.(fx,fy,(red,))
end

function run_loop(runner::DaggerRunner,loopranges,outbuffers...;groupspecs=nothing)
    @noinline run_loop(runner,runner.op,runner.inbuffers_pure,runner.loopranges,runner.workerthreads,
    runner.outars,runner.threaded,loopranges,outbuffers...;groupspecs)
end


function run_loop(::DaggerRunner,op,inbuffers_pure,runnerloopranges,workerthreads,
    outars,threaded, loopranges,outbuffers...;groupspecs=nothing)
    @debug "Groupspecs are ", groupspecs
    piddir = if groupspecs !== nothing && any(i->in(:output_chunk,i.reasons),groupspecs)
        tempname()
    else
        nothing
    end
    @debug "Pidddir is $piddir"
    local_outbuffers = make_outbuffer_shard(op,runnerloopranges,workerthreads)
    op = op
    r = broadcast(loopranges) do inow
        Dagger.spawn(inbuffers_pure,local_outbuffers,inow,piddir,outars,loopranges) do inbuffers_pure, outbuffers, inow, piddir, outars,loopranges
            @debug myid(), " Starting block ", inow
            inbuffers_wrapped = read_range.((inow,),op.inars,inbuffers_pure);
            outbuffers_now = extract_outbuffer.((inow,),op.outspecs,op.f.init,op.f.buftype,outbuffers)
            run_block(op,inow,inbuffers_wrapped,outbuffers_now,threaded)
            @debug myid(), "Finished running block ", inow

            put_buffer.((inow,),op.f.finalize, outbuffers_now, outbuffers, outars, (piddir,))
            true
        end
    end

    @debug myid(), " Finished spawning jobs"
    all(fetch.(r)) || error("Error during chunk processing")
    @debug myid(), " Fetched everything"
    if (groupspecs !== nothing) && any(i->in(:reducedim,i.reasons),groupspecs)
        @debug "Merging buffers"
        procs = unique(Dagger.processor.(fetch.(r,raw=true)))
        @debug "Affected processors are $procs"
        red = op.f.red
        buffers_used = collect(v for (k,v) in local_outbuffers.chunks if any(p->matches_proc(k,p),procs))
        @debug "Merging buffers from $(length(buffers_used)) workers."
        buffers_used = fetch.(buffers_used)
        collections_merged = merge_all_outbuffers(buffers_used,op.f.red)
        @debug "Writing merged buffers $(typeof(collections_merged))"
        unflushed_buffers = Dagger.spawn(collections_merged,op.f.finalize,outars,piddir) do cm,fin,outars,pdir
            flush_all_outbuffers(cm,fin,outars,pdir)
        end
        if !isempty(outbuffers)
            outbuffers = last(outbuffers)
            @debug "Putting back flushed buffers"
            r = Dagger.spawn(unflushed_buffers,outbuffers,red) do rembuf,outbuf, red
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
        run_group(sch,nothing)
    end
    true
end

function schedule(sch::DiskEngineScheduler,r::DaggerRunner,loopdims,loopsub,groupspecs)
    @debug "Starting to schedule: "
    r = map(loopsub) do i
        lrsub = subset_loopranges(sch.loopranges,loopdims,i.I)
        @debug "New split loopranges are: ", lrsub.members
        schsub = DiskEngineScheduler(sch.groups,lrsub,sch.runner)
        @debug "Spawning"
        outbuffers = make_outbuffer_shard(r.op,r.loopranges,r.workerthreads)
        Dagger.spawn(schsub,groupspecs,outbuffers) do sched, gs, ob
            run_group(sched,gs,ob)
        end
    end
    wait.(r)
end