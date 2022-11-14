using Distributed: @spawn, AbstractWorkerPool
using Base.Cartesian

updatears(clist, r, f, caches) =
    foreach(clist, caches) do ic, ca
        if !has_window(ic)
            updatear(f, r, ic.cube, geticolon(ic), ic.loopinds, ca)
        else
            updatear_window(
                r,
                ic.cube,
                geticolon(ic),
                ic.loopinds,
                ca,
                zip(ic.iwindow, ic.window),
                getwindowoob(ic),
            )
        end
    end
getindsall(indscol, loopinds, rfunc, colfunc = _ -> Colon()) =
    getindsall((), 1, (sort(indscol)...,), (loopinds...,), rfunc, colfunc)
function getindsall(indsall, inow, indscol, loopinds, rfunc, colfunc)
    if !isempty(indscol) && first(indscol) == inow
        getindsall(
            (indsall..., colfunc(inow)),
            inow + 1,
            Base.tail(indscol),
            loopinds,
            rfunc,
            colfunc,
        )
    else
        getindsall(
            (indsall..., rfunc(first(loopinds))),
            inow + 1,
            indscol,
            Base.tail(loopinds),
            rfunc,
            colfunc,
        )
    end
end
getindsall(indsall, inow, ::Tuple{}, ::Tuple{}, r, c) = indsall

function updatear(r, cube, cache,init)
    data = getdata(cube)
    
    hinds = first.(oo)
    indsall2 = last.(oo)
    fill!(cache, windowoob)
    cache[hinds...] = data[indsall2...]
end

function updatear(f, r, cube, indscol, loopinds, cache)
    indsall = getindsall(indscol, loopinds, i -> r[i])
    l2 = map((i, s) -> isa(i, Colon) ? s : length(i), indsall, size(cache))
    if size(cache) != l2
        hinds = map((i, s) -> isa(i, Colon) ? (1:s) : 1:length(i), indsall, size(cache))
        if f == :read
            cache[hinds...] = getdata(cube)[indsall...]
        else
            getdata(cube)[indsall...] = cache[hinds...]
        end
    else
        if f == :read
            d = getdata(cube)[indsall...]
            cache[:] = d
        else
            _writedata(getdata(cube), cache, indsall)
        end
    end
end
_writedata(d,cache,indsall) = d[indsall...] = cache
_writedata(d::Array{<:Any,0},cache::Array{<:Any,0},::Tuple{}) = d[] = cache[]


updateinars(dc, r, incaches) = updatears(dc.incubes, r, :read, incaches)
writeoutars(dc, r, outcaches) = updatears(dc.outcubes, r, :write, outcaches)

function pmap_with_data(f, p::AbstractWorkerPool, c...; initfunc, progress=nothing, kwargs...)
    d = Dict(ip=>remotecall(initfunc, ip) for ip in workers(p))
    allrefs = @spawn d
    function fnew(args...,)
        refdict = fetch(allrefs)
        myargs = fetch(refdict[myid()])
        f(args..., myargs)
    end
    if progress !==nothing
        progress_pmap(fnew,p,c...;progress=progress,kwargs...)
    else
        pmap(fnew,p,c...;kwargs...)
    end
end
pmap_with_data(f,c...;initfunc,kwargs...) = pmap_with_data(f,default_worker_pool(),c...;initfunc,kwargs...) 

function moduleloadedeverywhere()
    try
        isloaded = map(workers()) do w
            #We try calling a function defined inside this module, thi will error when YAXArrays is not loaded on the remote workers
            remotecall(() -> true, w)
        end
        fetch.(isloaded)
    catch e
        return false
    end
    return true
end

function runLoop(ec::EngineConfig, showprog)
    allRanges = GridChunks(getloopchunks(ec)...)
    if ec.ispar
        #Test if YAXArrays is loaded on all workers:
        moduleloadedeverywhere() || error(
            "YAXArrays is not loaded on all workers. Please run `@everywhere using YAXArrays` to fix.",
        )
        dcref = @spawn ec
        prepfunc = ()->getallargs(fetch(dcref))
        prog = showprog ? Progress(length(allRanges)) : nothing
        pmap_with_data(allRanges, initfunc=prepfunc, progress=prog) do r, prep
            incaches, outcaches, args = prep
            updateinars(dc, r, incaches)
            innerLoop(r, args...)
            writeoutars(dc, r, outcaches)
        end
    else
        incaches, outcaches, args = getallargs(dc)
        mapfun = showprog ? progress_map : map
        mapfun(allRanges) do r
            updateinars(dc, r, incaches)
            innerLoop(r, args...)
            writeoutars(dc, r, outcaches)
        end
    end
    dc.outcubes
end

function allocatecachebuf(ic, loopcachesize)
    s = size(ic.cube)
    indsall = getindsall(geticolon(ic), ic.loopinds, i -> loopcachesize[i], i -> s[i])
    if has_window(ic)
        indsall = Base.OneTo.(indsall)
        for (iw, (pre, after)) in zip(ic.iwindow, ic.window)
            old = indsall[iw]
            new = (first(old)-pre):(last(old)+after)
            indsall = Base.setindex(indsall, new, iw)
        end
        #@show indsall
        OffsetArray(Array{eltype(ic.cube)}(undef, length.(indsall)...), indsall...)
    else
        Array{eltype(ic.cube)}(undef,indsall...)
    end
end

function innercode(
    cI,
    f,
    xinBC,
    xoutBC,
)
    #Copy data into work arrays
    myinwork = map(xinBC) do x
        view(x, cI.I...)
    end
    myoutwork = map(xoutBC) do x
        view(x, cI.I...)
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

@noinline function innerLoop(loopRanges,args...)
    for cI in CartesianIndices(map(i -> 1:length(i), loopRanges))
        innercode(cI,args...)
    end
end

@noinline function innerLoop_threaded(loopRanges,args...)
    Threads.@threads for cI in CartesianIndices(map(i -> 1:length(i), loopRanges))
        innercode(cI,args...)
    end
end