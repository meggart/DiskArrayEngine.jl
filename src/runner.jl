using Distributed: @spawn, AbstractWorkerPool
using ProgressMeter: Progress, next!
using Base.Cartesian

#We do not make a view when accessing single values
@inline _view(::Input, a,i::Int...) = a[i...]
@inline _view(::Any, a,i...) = view(a,i...)
@inline apply_offset(window,offset) = window .- offset


@inline function _view(x::ArrayBuffer,I,io)
    ms = mysub(x,I)
    windowsub = inner_getindex(x.lw.windows,ms)
    inds = apply_offset.(windowsub,x.offsets)
    _view(io, x.a,inds...)
end

@inline inner_getindex(p::ProductArray,I) = inner_getindex.(p.members,I)
inner_getindex(w,i::Int) = w[i]

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
    apply_function(f,myoutwork, myinwork)
    nothing
end

function run_block(f,inow,inbuffers_wrapped,outbuffers_now,lspl)
    if lspl === nothing
        run_block_single(inow, f, inbuffers_wrapped, outbuffers_now)
    else
        @debug "Using $lspl to split loops"
        run_block_threaded(inow, lspl,f, inbuffers_wrapped, outbuffers_now)
    end
end

@noinline function run_block_single(loopRanges,f, inbuffers, outbuffers)
    for cI in CartesianIndices(loopRanges)
       innercode(cI,f,inbuffers, outbuffers)
    end
end

@noinline function run_block_threaded(loopRanges,lspl,f,inbuffers,outbuffers)
    tri, ntri = split_loopranges_threads(lspl,loopRanges)
    if isempty(tri)
        run_block_single(loopRanges,f,inbuffers,outbuffers)
    else
        if isempty(ntri)
            Threads.@threads for i_thread in CartesianIndices(tri)
                innercode(i_thread,f,inbuffers,outbuffers)
            end
        else
            for i_nonthread in CartesianIndices(ntri)
                Threads.@threads for i_thread in CartesianIndices(tri)
                    cI = merge_loopranges_threads(i_thread,i_nonthread,lspl)
                    innercode(cI,f,inbuffers,outbuffers)
                end
            end
        end
    end
end

function run_block(op::GMDWop,loopRanges,xin,xout,threaded) 
    lspl = if threaded && op.f.allow_threads
        op.lspl
    else
        nothing
    end
    run_block(op.f.f,loopRanges,xin,xout,lspl)
end

function run_block(f::BlockFunction,loopRanges,xin,xout,lspl)
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
    _run_block(f,myinwork,myoutwork,isa(lspl,LoopIndSplitter))
end
function _run_block(f::BlockFunction{<:Any,Mutating},myinwork,myoutwork,threaded::Bool)
    f.f(myoutwork...,myinwork...,dims=getdims(f),threaded=threaded)
end
function _run_block(f::BlockFunction{<:Any,NonMutating},myinwork,myoutwork,threaded::Bool)
    r = f.f(myinwork...,dims=getdims(f),threaded=threaded)
    map(myoutwork,r) do o,ir
        o .= ir
    end
end

plan_to_loopranges(lr) = lr
plan_to_loopranges(lr::ExecutionPlan) = lr.lr

struct LocalRunner
    op
    loopranges
    outars
    threaded
    inbuffers_pure
    outbuffers
    progress
    restarter
end
function LocalRunner(op,exec_plan,
    outars=create_outars(op,exec_plan);
    threaded=true,
    showprogress=true,
    restartfile=nothing,
    restartmode=:continue,
    )
    loopranges = plan_to_loopranges(exec_plan)
    inbuffers_pure = generate_inbuffers(op.inars, loopranges)
    outbuffers = generate_outbuffers(op.outspecs,op.f, loopranges)
    pm = showprogress ? Progress(length(loopranges)) : nothing
    loopranges = plan_to_loopranges(exec_plan)
    restarter = create_restarter(restartfile,loopranges,restartmode)
    LocalRunner(op,loopranges,outars, threaded, inbuffers_pure,outbuffers,pm,restarter)
end

update_progress!(::Nothing) = nothing
update_progress!(pm) = next!(pm)
need_run(inow,restarter::Restarter) = need_run(inow,restarter.remaining_loopranges)
need_run(inow,::Nothing) = true
need_run(inow,remaining_loopranges) = inow in remaining_loopranges


function run_loop(runner::LocalRunner,loopranges = runner.loopranges;groupspecs=nothing)
    run_loop(
        runner,runner.op, runner.inbuffers_pure,runner.outbuffers,runner.threaded,runner.outars,runner.progress,loopranges, runner.restarter;groupspecs
    )
end

function default_loopbody(inow, re, op, inbuffers_pure, outbuffers, threaded,outars, progress)
    @debug "inow = ", inow
    if need_run(inow,re)
        inbuffers_wrapped = read_range.((inow,),op.inars,inbuffers_pure);
        outbuffers_now = extract_outbuffer.((inow,),op.outspecs,op.f.init,op.f.buftype,outbuffers)
        run_block(op,inow,inbuffers_wrapped,outbuffers_now,threaded)
        put_buffer.((inow,),outbuffers_now,outars,nothing)
        clean_aggregator.(outbuffers)
        update_progress!(progress)
        update_restarter(re, inow)
    end
end

@noinline function run_loop(::LocalRunner,op,inbuffers_pure,outbuffers,threaded,outars,progress,loopranges,re;groupspecs=nothing)
    for inow in loopranges
        default_loopbody(inow, re, op, inbuffers_pure, outbuffers, threaded,outars, progress)
    end
    finish_progress(progress)
    finish_restarter(re)
end

using Distributed

struct PMapRunner
    op
    loopranges
    outars
    threaded
    inbuffers_pure
    outbuffers
    progress_channel
    restarter
end
function PMapRunner(op,exec_plan,outars=create_outars(op,exec_plan);threaded=true,showprogress=true,restartfile=nothing,restartmode=:continue)  
    all(isnothing,op.f.red) || error("PMapRunner can not be used for reductions. Use DaggerRunner instead")
    loopranges = plan_to_loopranges(exec_plan)
    inbuffers_pure = generate_inbuffers(op.inars, loopranges)
    outbuffers = generate_outbuffers(op.outspecs,op.f, loopranges)
    progress_channel = if showprogress
        progress = Progress(length(loopranges))
        channel = Distributed.RemoteChannel(()->Channel{Bool}(), 1)
        @async while take!(channel)
            next!(progress)
        end
        channel
    else
        nothing
    end
    restarter = create_restarter(restartfile,loopranges,restartmode)
    restart_channel = if isnothing(restarter)
        nothing
    else
        nd = ndims(restarter)
        channel = Distributed.RemoteChannel(()->Channel{Union{Nothing,NTuple{nd,Int}}}(), 1)
        @async while true
            update = take!(channel)
            isnothing(update) && break
            add_entry(restarter,update)
        end
    end
    PMapRunner(op,loopranges, outars, threaded, inbuffers_pure,outbuffers,progress_channel,restart_channel)
end

update_progress!(pm::RemoteChannel) = put!(pm, true)
update_restarter(re::RemoteChannel,i) = put!(re, i)
update_restarter(::Nothing,i) = nothing
update_restarter(re::Restarter,i) = add_entry(re,i)
finish_progress(::Any) = nothing
finish_progress(pm::RemoteChannel) = put!(pm,false)
finish_restarter(::Any) = nothing
finish_restarter(re::RemoteChannel) = put!(re,nothing)

function run_loop(runner::PMapRunner,loopranges = runner.loopranges;groupspecs=nothing)
    run_loop(
        runner,runner.op, runner.inbuffers_pure,runner.outbuffers,runner.threaded,runner.outars,runner.progress_channel,runner.restarter,loopranges;groupspecs
    )
end

@noinline function run_loop(::PMapRunner,op,inbuffers_pure,outbuffers,threaded,outars,progress,restarter,loopranges;groupspecs=nothing)
    pmap(CachingPool(workers()),loopranges) do inow
        default_loopbody(inow, restarter, op, inbuffers_pure, outbuffers, threaded,outars, progress)
    end
    finish_progress(progress)
end

