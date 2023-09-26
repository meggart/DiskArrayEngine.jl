using Distributed

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
      ioverlap = findall(i->!isa(get_overlap(i),NonOverlapping),spec.lw.windows.members)
      union!(setdiff(alld,li),ioverlap)
    end
    allreddims = reduce(union!,outreduceinds,init=Int[])
    @debug "Reducedims are $allreddims"
    LoopIndSplitter(nd,(allreddims...,))
  end

using Distributed
mutable struct WorkerData
  i::Int
  data::RemoteChannel
  function WorkerData(i,data)
    c = RemoteChannel(i)
    put!(c,data)
    w = new(i,c)
    finalizer(clear!,w)
    w
  end
end
clear!(w::WorkerData) = finalize(w.data)
getdata(w::WorkerData) = fetch(w.data)

struct WorkerDataPool
  channel::Channel{Int}
  workers::Set{WorkerData}
end


  # mutable struct DataPool <: AbstractWorkerPool
  #   channel::Channel{Int}
  #   workers::Set{Int}
  #   data
  #   # Mapping between a worker_id and a RemoteChannel
  #   map_objects::Dict{Int, RemoteChannel}
  
  #   function DataPool(data)
  #       wp = new(Channel{Int}(typemax(Int)), Set{Int}(), data, Dict{Int, RemoteChannel}())
  #       finalizer(clear!, wp)
  #       wp
  #   end
  #   #Here we assume there is already an dict, no additional finalizer is added
  #   function DataPool(data,workers::Set{Int},objdict)
  #     new(Channel{Int}(typemax(Int)), workers, data, objdict)
  #   end
  # end
  
  # #serialize(::Base.AbstractSerializer, ::DataPool) = throw(ErrorException("DataPool objects are not serializable."))
  
  # function DataPool(workers::Vector{Int},data)
  #   pool = DataPool(data)
  #   for w in workers
  #       push!(pool, w)
  #   end
  #   return pool
  # end
  
  # function Distributed.clear!(pool::DataPool)
  #   for (_,rr) in pool.map_objects
  #       finalize(rr)
  #   end
  #   empty!(pool.map_objects)
  #   pool
  # end
  
  # exec_from_data(rr::RemoteChannel, f, args...; kwargs...) = f(fetch(rr),args...; kwargs...)
  # function exec_from_data(r::Tuple{<:Any, RemoteChannel}, f, args...; kwargs...)
  #   data =   r[1]()
  #   put!(r[2], data)        # Cache locally
  #     f(data, args...; kwargs...)
  # end

  # "Subset a data pool while retaining the same cached data"
  # function subsetpool(p::DataPool,workers)
  #   for w in

  #   end
  # end
  
  # function Distributed.remotecall_pool(rc_f, f, pool::DataPool, args...; kwargs...)
  #   worker = take!(pool)
  #   r = if !haskey(pool.map_objects,worker)
  #     c = RemoteChannel(worker)
  #     pool.map_objects[worker] = c
  #     (pool.data,c)
  #   else
  #     pool.map_objects[worker]
  #   end
  #   try
  #       rc_f(exec_from_data, worker, r, f, args...; kwargs...)
  #   finally
  #       put!(pool, worker)
  #   end
  # end