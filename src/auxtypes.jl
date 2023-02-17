"Type very similar to `Iterators.product`, but for indexable arrays."
struct ProductArray{T,N,S<:Tuple} <:AbstractArray{T,N}
    members::S
end
function ProductArray(members::Tuple)
    et = Tuple{eltype.(members)...}
    ProductArray{et,length(members),typeof(members)}(members)
end
Base.size(a::ProductArray,i::Int) = length(a.members[i])
Base.size(a::ProductArray) = length.(a.members)
Base.axes(a::ProductArray) = first.(axes.(a.members))
Base.IndexStyle(::Type{<:ProductArray})=Base.IndexCartesian()
Base.getindex(a::ProductArray{<:Any,N}, i::Vararg{Int, N}) where N = getindex.(a.members,i)

struct RegularWindows <: AbstractVector{UnitRange{Int}}
    start::Int
    stop::Int
    step::Float64
    window::Int
  end
  RegularWindows(start,stop;window=1,step=window) = RegularWindows(start,stop,step,window)
  Base.size(r::RegularWindows) = (ceil(Int,(r.stop-r.start+1)/r.step),)
  function Base.getindex(r::RegularWindows, i::Int)
    i0 = r.start + floor(Int,(i-1)*r.step)
    i0 > r.stop && throw(BoundsError(r,i))
    i0:min((i0+r.window-1),r.stop)
  end

struct MovingWindow <: AbstractVector{UnitRange{Int}}
  first::Int
  steps::Int
  width::Int
  n::Int
end
Base.size(m::MovingWindow) = (m.n,)
Base.getindex(m::MovingWindow,i::Int) = (m.first+(i-1)*m.steps):(m.first+(i-1)*m.steps+m.width-1)
  

using Distributed

"""
Very similar to `pmap` from Distributed. However, in addition one passes an `initfunc` that does some
initial work on every worker (loading common data etc...). This result of the call to `initfunc` will
be appended as the last argument to every function call. 
"""
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

"Type used for dispatch to show something is done in input mode"
struct Input end

"Type used for dispatch to show something is done in output mode"
struct Output end


struct LoopIndSplitter{TR,NTR,BACK}
end
threadinds(::LoopIndSplitter{TR},lr) where TR = map(Base.Fix1(getindex,lr),TR)
nonthreadinds(::LoopIndSplitter{<:Any,NTR},lr) where NTR = map(Base.Fix1(getindex,lr),NTR)
get_back(::LoopIndSplitter{<:Any,<:Any,B}) where B = B
function LoopIndSplitter(nd,reddims::Tuple)
  nonreddims = (setdiff(1:nd,reddims)...,)
  back = ntuple(nd) do i
    ir = findfirst(==(i),reddims)
    if ir !== nothing
      return (false,ir)
    else
      ir = findfirst(==(i),nonreddims)
      return (true,ir)
    end
  end
  LoopIndSplitter{nonreddims,reddims,back}()
end
split_loopranges_threads(lspl,lr) = threadinds(lspl,lr),nonthreadinds(lspl,lr)
function merge_loopranges_threads(i_tr::CartesianIndex,i_ntr::CartesianIndex,lspl)
  b = get_back(lspl)
  map(b) do (is_tr,i)
    if is_tr
      i_tr.I[i]
    else
      i_ntr.I[i]
    end
  end |> CartesianIndex
end
get_loopsplitter(op) = get_loopsplitter(length(op.windowsize),op.outspecs)
function get_loopsplitter(nd,outspecs)
    alld = 1:nd
    outreduceinds = map(outspecs) do spec
      li = getloopinds(spec)
      setdiff(alld,li)
    end
    allreddims = reduce(union!,outreduceinds,init=Int[])
    LoopIndSplitter(nd,(allreddims...,))
  end