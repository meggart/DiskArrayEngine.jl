using Distributed: @spawn, AbstractWorkerPool
using Base.Cartesian


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

function runLoop(ec::EngineConfig, showprog, use_dist)
    allRanges = GridChunks(getloopchunks(ec)...)
    if use_dist
        #Test if YAXArrays is loaded on all workers:
        moduleloadedeverywhere() || error(
            "YAXArrays is not loaded on all workers. Please run `@everywhere using YAXArrays` to fix.",
        )
        dcref = @spawn ec
        prepfunc = ()->getallargs(fetch(dcref))
        prog = showprog ? Progress(length(allRanges)) : nothing
        pmap_with_data(allRanges, initfunc=prepfunc, progress=prog) do r, prep
            incaches, outcaches, args = prep
            updateinars(ec, r, incaches)
            run_block(r, args...)
            writeoutars(ec, r, outcaches)
        end
    else
        incaches, outcaches, args = getallargs(dc)
        mapfun = showprog ? progress_map : map
        mapfun(allRanges) do r
            updateinars(ec, r, incaches)
            run_block(r, args...)
            writeoutars(ec, r, outcaches)
        end
    end
    ec.outcubes
end


#We do not make a view when accessing single values
@inline _view(::Input, a,i::Int...) = a[i...]
@inline _view(::Any, a,i...) = view(a,i...)
@inline apply_offset(window,offset) = window.-offset


@inline function _view(x::ArrayBuffer,I,io)
    inds = apply_offset.(x.lw.windows[mysub(x,I)...],x.offsets)
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

@noinline function run_block(loopRanges,args...)
    for cI in CartesianIndices(loopRanges)
        innercode(cI,args...)
    end
end

@noinline function run_block_threaded(loopRanges,args...)
    Threads.@threads for cI in CartesianIndices(loopRanges)
        innercode(cI,args...)
    end
end

function run_loop(op, loopranges,outars)

    inbuffers_pure = generate_inbuffers(op.inars, loopranges)
  
    outbuffers = generate_outbuffers(outars,f, loopranges)
  
    for inow in loopranges
      @show inow
      inbuffers_wrapped = read_range.((inow,),inars,inbuffers_pure);
      outbuffers_now = wrap_outbuffer.((inow,),outars,(f,),outbuffers)
      DiskArrayEngine.run_block(inow, f, inbuffers_wrapped, outbuffers_now)
    
      put_buffer.((inow,), (f,), outbuffers_now, outbuffers, outars)
    end
  end
  