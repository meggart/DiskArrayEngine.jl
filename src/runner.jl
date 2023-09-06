using Distributed: @spawn, AbstractWorkerPool
using ProgressMeter: Progress, next!
using Base.Cartesian

#We do not make a view when accessing single values
@inline _view(::Input, a,i::Int...) = a[i...]
@inline _view(::Any, a,i...) = view(a,i...)
@inline apply_offset(window,offset) = window .- offset


@inline function _view(x::ArrayBuffer,I,io)
    ms = mysub(x,I)
    windowsub = x.lw.windows[ms...]
    inds = apply_offset.(windowsub,x.offsets)
    _view(io, x.a,inds...)
end



function innercode(
    cI,
    f,
    xin,
    xout,
    )
    #Copy data into work arrays
    myinwork = map(xin) do x
        _view(x, cI.I, Input())
    end
    myoutwork = map(xout) do x
        _view(x, cI.I, Output())
    end
    #Apply filters
    mvs = applyfilter(f,myinwork)
    if any(mvs)
        # Set all outputs to missing
        foreach(ow -> fill!(ow, missing), myoutwork)
    else
        #Finally call the function
        apply_function(f,myoutwork, myinwork)
    end
end

function run_block(op,inow,inbuffers_wrapped,outbuffers_now,threaded)
    if !threaded
        run_block_single(inow, op.f, inbuffers_wrapped, outbuffers_now)
    else
        lspl = get_loopsplitter(op)
        @debug "Using $lspl to split loops"
        run_block_threaded(inow, lspl,op.f, inbuffers_wrapped, outbuffers_now)
    end
end

function run_block_single(loopRanges,f::UserOp,args...)
    Threads.@threads for cI in CartesianIndices(loopRanges)
       innercode(cI,f,args...)
    end
end

@noinline function run_block_threaded(loopRanges,lspl,f::UserOp,args...)
    tri, ntri = split_loopranges_threads(lspl,loopRanges)
    for i_nonthread in CartesianIndices(ntri)
        Threads.@threads for i_thread in CartesianIndices(tri)
            cI = merge_loopranges_threads(i_thread,i_nonthread,lspl)
            innercode(cI,f,args...)
        end
    end
end

function run_block(f::GMDWop{<:Any,<:Any,<:Any,<:UserOp{<:BlockFunction}},loopRanges,xin,xout,threaded)
    i1 = first.(loopRanges)
    i2 = last.(loopRanges)
    myinwork = map(xin) do x
        firsti = first.(x.lw.windows[mysub(x,i1)...])
        lasti = last.(x.lw.windows[mysub(x,i2)...])
        iw1 = apply_offset.(firsti,x.offsets)
        iw2 = apply_offset.(lasti,x.offsets)
        rr = range.(iw1,iw2)
        OffsetArray(view(x.a, rr...),firsti .- 1)
    end
    myoutwork = map(xout) do x
        firsti = first.(x.lw.windows[mysub(x,i1)...])
        lasti = last.(x.lw.windows[mysub(x,i2)...])
        iw1 = apply_offset.(firsti,x.offsets)
        iw2 = apply_offset.(lasti,x.offsets)
        rr = Base.IdentityUnitRange.(range.(iw1,iw2))
        OffsetArray(view(x.a, rr...),firsti .- 1)
    end
    _run_block(f.f,myinwork,myoutwork,threaded)
end
function _run_block(f::UserOp{<:BlockFunction{<:Any,Mutating}},myinwork,myoutwork,threaded)
    f.f.f(myoutwork...,myinwork...,f.args...;f.kwargs...,dims=getdims(f.f),threaded=threaded)
end
function _run_block(f::UserOp{<:BlockFunction{<:Any,NonMutating}},myinwork,myoutwork,threaded)
    r = f.f.f(myinwork...,f.args...;f.kwargs...,dims=getdims(f.f),threaded=threaded)
    map(myoutwork,r) do o,ir
        o .= ir
    end
end

plan_to_loopranges(lr) = lr
plan_to_loopranges(lr::ExecutionPlan) = lr.lr

struct LocalRunner{OP,LR,OA,IB,OB,P}
    op::OP
    loopranges::LR
    outars::OA
    threaded::Bool
    inbuffers_pure::IB
    outbuffers::OB
    progress::P
end
function LocalRunner(op,exec_plan,outars;threaded=true,showprogress=true)
    loopranges = plan_to_loopranges(exec_plan)
    inbuffers_pure = generate_inbuffers(op.inars, loopranges)
    outbuffers = generate_outbuffers(op.outspecs,op.f, loopranges)
    pm = showprogress ? Progress(length(loopranges)) : nothing
    LocalRunner(op,plan_to_loopranges(loopranges),outars, threaded, inbuffers_pure,outbuffers,pm)
end

update_progress!(::Nothing) = nothing
update_progress!(pm) = next!(pm)

function run_loop(runner::LocalRunner,loopranges = runner.loopranges;groupspecs=nothing)
    for inow in loopranges
        @debug "inow = ", inow
        inbuffers_wrapped = read_range.((inow,),runner.op.inars,runner.inbuffers_pure);
        outbuffers_now = extract_outbuffer.((inow,),runner.op.outspecs,runner.op.f.init,runner.op.f.buftype,runner.outbuffers)
        run_block(runner.op,inow,inbuffers_wrapped,outbuffers_now,runner.threaded)
        put_buffer.((inow,),runner.op.f.finalize, outbuffers_now, runner.outbuffers, runner.outars,nothing)
        update_progress!(runner.progress)
    end
end


struct DistributedRunner{OP,LR,OA,IB,OB}
    op::OP
    loopranges::LR
    outars::OA
    threaded::Bool
    inbuffers_pure::IB
    outbuffers::OB
    workers::CachingPool
end
function DistributedRunner(op,loopranges,outars;threaded=true,w = workers())
    inars = op.inars
    makeinbuf = ()->begin
        generate_inbuffers(inars, loopranges)
    end
    outspecs = op.outspecs
    f = op.f
    makeoutbuf = ()->begin
        generate_outbuffers(outspecs,f, loopranges)
    end
    allinbuffers = makeinbuf
    alloutbuffers = makeoutbuf

    DistributedRunner(op,loopranges,outars, threaded, allinbuffers,alloutbuffers,CachingPool(w))
end


# function DiskArrayEngine.run_loop(runner::DistributedRunner,loopranges = runner.loopranges;groupspecs=nothing)
#     if groupspecs !== nothing && :output_chunk in groupspecs
#         piddir = @spawn tempname()
#     else
#         piddir = nothing
#     end
#     cpool = runner.workers
#     op = runner.op
#     threaded = runner.threaded
#     pmap(cpool,loopranges) do stored, inow
#     #map(loopranges) do inow
#         println("inow = ", inow)
#         @debug "inow = ", inow

#         inbuffers_pure,outbuffers = fetch(inbuf[myid()]),fetch(outbuf[myid()])
#         inbuffers_wrapped = read_range.((inow,),op.inars,inbuffers_pure);
        
#         outbuffers_now = wrap_outbuffer.((inow,),op.outspecs,op.f.init,op.f.buftype,outbuffers)
#         run_block(op,inow,inbuffers_wrapped,outbuffers_now,threaded)

#         put_buffer.((inow,),op.f.finalize, outbuffers_now, outbuffers, outars, (piddir,))

#     end
#     if groupspecs !== nothing && :reducedim in groupspecs
#         @debug "Merging buffers"
#         collection_merged = foldl(values(runner.outbuffers)) do agg,buffers
#             buf = fetch(buffers)
#             merge_outbuffer_collection.(agg,buf)
#         end
#         @debug "Writing merged buffers"
#         map(collection_merged) do outbuffer
#             for (k,v) in outbuffer.buffers
#                 @debug "Writing index $k"
#                 put_buffer.((k,),runner.op.f.finalize, v, outbuffer, runner.outars, (piddir,))
#             end
#         end
#     end
# end

