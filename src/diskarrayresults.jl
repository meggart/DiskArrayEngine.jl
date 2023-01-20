using DiskArrays: AbstractDiskArray, RegularChunks
using OffsetArrays: OffsetArray

struct GMWOPResult{T,N,G<:GMDWop,CS,ISPEC} <: AbstractDiskArray{T,N}
    op::G
    ires::Val{ISPEC}
    chunksize::CS
    max_cache::Float64
    s::NTuple{N,Int}
  end
  getoutspec(r::GMWOPResult{<:Any,<:Any,<:Any,<:Any,ISPEC}) where ISPEC = r.op.outspecs[ISPEC]
  getioutspec(::GMWOPResult{<:Any,<:Any,<:Any,<:Any,ISPEC}) where ISPEC = ISPEC
  
  Base.size(r::GMWOPResult) = length.(getoutspec(r).windows.members)
  
  function results_as_diskarrays(o::GMDWop;cs=nothing,max_cache=1e9)
    map(enumerate(o.outspecs)) do (i,outspec)
      T = o.f.outtype[i]
      N = ndims(outspec.windows)
      cs = cs === nothing ? DiskArrays.Unchunked() : cs
      GMWOPResult{T,N,typeof(o),typeof(cs),i}(o,Val(i),cs,max_cache,size(outspec.windows)) 
    end
  end
  
  
  function DiskArrays.readblock!(res::GMWOPResult, aout,r::AbstractUnitRange...)
    #Find out directly connected loop ranges
    s = res.op.windowsize
    s = Base.OneTo.(s)
    outars = ntuple(_->nothing,length(res.op.outspecs))
    outspec = getoutspec(res)
    outars = Base.setindex(outars,InputArray(OffsetArray(aout,r...),outspec),getioutspec(res))
    foreach(getloopinds(outspec),r) do li,ri
      s = Base.setindex(s,ri,li)
    end
    l = length.(s)
    lres = mysub(outspec,s)  
    if length(lres) < length(l) && prod(l)*sizeof(eltype(res)) > res.max_cache
      l = cut_looprange(l,res.max_cache)
    end
    loopranges = ProductArray(map(s,l) do si,cs
      ProcessingSteps(1-first(si),RegularChunks(cs,0,length(si)))
    end)
    run_loop(res.op,loopranges,outars)
    nothing
  end