export results_as_diskarrays
using DiskArrays: AbstractDiskArray, RegularChunks
using OffsetArrays: OffsetArray

struct GMWOPResult{T,N,G<:GMDWop,CS,ISPEC} <: AbstractEngineArray{T,N}
    op::G
    ires::Val{ISPEC}
    chunksize::CS
    max_cache::Float64
    s::NTuple{N,Int}
  end
  getoutspec(r::GMWOPResult{<:Any,<:Any,<:Any,<:Any,ISPEC}) where ISPEC = r.op.outspecs[ISPEC]
  getioutspec(::GMWOPResult{<:Any,<:Any,<:Any,<:Any,ISPEC}) where ISPEC = ISPEC
  
  Base.size(r::GMWOPResult) = maximum.(windowmax,getoutspec(r).lw.windows.members)
  
  function results_as_diskarrays(o::GMDWop;cs=nothing,max_cache=1e9)
    ntuple(length(o.outspecs)) do i
      outspec = o.outspecs[i]
      T = o.f.outtype[i]
      N = ndims(outspec.lw.windows)
      cs = cs === nothing ? DiskArrays.Unchunked() : cs
      GMWOPResult{T,N,typeof(o),typeof(cs),i}(o,Val(i),cs,max_cache,size(outspec.lw.windows)) 
    end
  end
  
  
  function DiskArrays.readblock!(res::GMWOPResult, aout,r::AbstractUnitRange...)
    #Find out directly connected loop ranges
    s = res.op.windowsize
    s = Base.OneTo.(s)
    outars = ntuple(_->nothing,length(res.op.outspecs))
    outspec = getoutspec(res)
    foreach(getloopinds(outspec),r,outspec.lw.windows.members) do li,ri,w
      i1 = findfirst(a->windowmax(a)>=first(ri),w)
      i2 = findlast(a->windowmin(a)<=last(ri),w)
      s = Base.setindex(s,i1:i2,li)
  end
    outars = Base.setindex(outars,OffsetArray(aout,r...),getioutspec(res))
    l = length.(s)
    lres = mysub(outspec.lw,s)
    if length(lres) < length(l) && prod(l)*DiskArrays.element_size(res) > res.max_cache
      l = cut_looprange(l,res.max_cache)
    end
    loopranges = map(s,l) do si,cs
      map(RegularChunks(cs,0,length(si)),first(si)-1) do c,offs
        c.+offs
      end
    end
    loopranges = ProductArray(loopranges)
    runner = LocalRunner(res.op,loopranges,outars)
    run_loop(runner,loopranges)
    nothing
  end

  function compute!(ret,a::DiskArrayEngine.GMWOPResult;runner=LocalRunner,threaded=true,max_cache=5e8,kwargs...)
    lr = DiskArrayEngine.optimize_loopranges(a.op,max_cache,tol_low=0.2,tol_high=0.05,max_order=2)
    par_only = runner <: DaggerRunner
    outars = create_outars(a.op,lr;par_only)
    iout = findfirst(i->Val(i)===a.ires,1:length(outars))
    if ret !== nothing
        outars = Base.setindex(outars,ret,iout)
    end
    r = runner(a.op,lr,outars;threaded,kwargs...)
    run(r)
    fetch(outars[iout])
end
function compute(a::DiskArrayEngine.GMWOPResult;runner=LocalRunner,threaded=true,kwargs...)
    compute!(nothing,a;runner,threaded,kwargs...)
end