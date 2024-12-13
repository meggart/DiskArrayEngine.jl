function is_output_chunk_overlap(spec,outar,idim,lr)
    li = getloopinds(spec)
    if idim in li
        ii = findfirst(==(idim),li)
        windows = spec.lw.windows.members[ii]
        isa(get_overlap(windows),NonOverlapping) || return false
        loopind = li[ii]
        cs = eachchunk(outar).chunks[ii]
        chunkbounds = cumsum(length.(cs))
        windows = spec.lw.windows.members[ii]
        looprange = lr.members[idim]
        length(looprange) == 1 && return false
        !all(looprange) do r
            # w1 = first(windows[first(r)])
            # w2 = last(windows[last(r)])
            w1 = inner_index(windows,first(r))
            w2 = inner_index(windows,first(r))
            cr = DiskArrays.findchunk(cs,first(w1):last(w2))
            #check if start and end are on a chunk boundary
            first(cs[first(cr)])==first(ii) && last(cs[last(cr)])==last(ii)
        end
    else
        false
    end
end
is_output_chunk_overlap(spec,::Nothing,idim,lr) = false
function is_output_reducedim(spec,outar,idim)
    li = getloopinds(spec)
    if in(idim,li)
        i = findfirst(==(idim),li)
        ov = get_overlap(spec.lw.windows.members[i])
        isa(ov,Repeating)
    else
        true
    end
end

function split_dim_reasons(op,lr,outars)
    ret = ntuple(_->Symbol[],ndims(lr))
    for (spec,ar) in zip(op.outspecs,outars)
        foreach(1:ndims(lr)) do idim
            if is_output_chunk_overlap(spec,ar,idim,lr)
                push!(ret[idim],:output_chunk)
            end
            if is_output_reducedim(spec,ar,idim)
                push!(ret[idim],:reducedim)
            end
        end
    end
    ret
end
reason_priority = Dict(
:foldl => 1, 
:reducedim => 2,
:output_chunk => 3, 
:overlapinputs =>4,
)

function get_procgroups(op, lr,outars)
    spr = split_dim_reasons(op,lr,outars)
    groups = GroupLoopDim[]
    
    while !all(isempty,spr)
        bestreason = argmin(s->isempty(s) ? 1000 : minimum(i->reason_priority[i],s),spr)
        dims = findall(==(bestreason),spr)
        
        push!(groups,GroupLoopDim((bestreason...,),(dims...,)))
        for d in dims
            empty!(spr[d])
        end
    end
    groups
end

struct GroupLoopDim
    reasons
    dims
end
struct DiskEngineScheduler{G,LR,R}
    groups::G
    loopranges::LR
    runner::R
end


function freeloopdims(sch::DiskEngineScheduler)
    nd = ndims(sch.loopranges)
    freedims = filter(i->length(sch.loopranges.members[i])>1,1:nd)
    groupdims = Int[]
    for g in sch.groups
        append!(groupdims,g.dims)
    end
    setdiff(freedims,groupdims)
end



function subset_loopranges(lr, dims, reps)
    mem = lr.members
    foreach(dims,reps) do d, r
        rsub = mem[d][r]
        mem = Base.setindex(mem,[rsub],d)
    end
    ProductArray(mem)
end

function schedule(sch::DiskEngineScheduler,::Any,loopdims,loopsub,groupspecs)
    for i in loopsub
        lrsub = subset_loopranges(sch.loopranges,loopdims,i.I)
        schsub = DiskEngineScheduler(sch.groups,lrsub,sch.runner)
        run_group(schsub,groupspecs)
    end
end

function run_group(sch, groupspecs,args...)
    #We just run everything if there are no groups left 
    @debug "Deciding how to run the group"
    if isempty(sch.groups)
        @debug "No subgroups available, stepping into run_loop"
        run_loop(sch.runner,sch.loopranges,args...;groupspecs)
    else
        loopdims = freeloopdims(sch)
        if !isempty(loopdims)
            loopsub = CartesianIndices((map(d->1:length(sch.loopranges.members[d]),loopdims)...,))
            @debug "Free loop dimensions available, splitting loop into $(length(loopsub)) subgroups"
            schedule(sch,sch.runner,loopdims,loopsub,groupspecs,args...)
        else 
            @debug "Groups are available for split"
            g = last(sch.groups)
            if groupspecs !== nothing
                g = (groupspecs...,g)
            else
                g = (g,)
            end
            @debug "New Group specs are ", g
            gnew = sch.groups[1:end-1]
            schnew = DiskEngineScheduler(gnew,sch.loopranges,sch.runner)
            run_group(schnew,g,args...)
        end
    end
end

function Base.run(runner::Union{LocalRunner,PMapRunner})
    groups = get_procgroups(runner.op, runner.loopranges, runner.outars)
    sch = DiskEngineScheduler(groups, runner.loopranges, runner)
    run_group(sch,nothing)
    return runner.outars
end



# function DiskArrayEngine.schedule(sch,::DistributedRunner,loopdims,loopsub,groupspecs)
#     w = workers(sch.runner.workers)
#     n_workers = length(w)
#     taskstorun = collect(loopsub)
#     tasksrunning = Dict{eltype(taskstorun),CachingPool}()
#     runningtlock = ReentrantLock()
#     workersavail = RemoteChannel(()->Channel{Int}(length(w)))
#     for iw in w
#         put!(workersavail,iw)
#     end
#     while !isempty(taskstorun)
#         iw = take!(workersavail)
#         @show "Got worker $iw"
#         i = popfirst!(taskstorun)
#         lrsub = subset_loopranges(sch.loopranges,loopdims,i.I)
#         newpool = DataPool([iw],sch.runner.workers.data)
#         newrunner = DistributedRunner(sch.runner.op,sch.runner.loopranges,sch.runner.outars,sch.runner.threaded,sch.runner.inbuffers_pure,sch.runner.outbuffers,newpool)
#         schsub = DiskArrayEngine.DiskEngineScheduler(sch.groups,lrsub,newrunner)
#         lock(runningtlock) do
#             tasksrunning[i] = newpool
#         end
#         begin
#             try
#                 println("Running group $i on loopdims $loopdims on worker $iw") 
#                 run_group(schsub;groupspecs,workerchannel = workersavail)
#                 println("Finished group $i on loopdims $loopdims on worker $iw")
#             # catch e
#             #     println(typeof(e))
#             finally
#                 lock(runningtlock) do
#                     delete!(tasksrunning,i)
#                 end            
#             end
#         end
#     end
# end

# function Base.run(runner::DistributedRunner)
#     groups = get_procgroups(runner.op, runner.loopranges, runner.outars)
#     sch = DiskEngineScheduler(groups, runner.loopranges, runner)
#     run_group(sch)
# end