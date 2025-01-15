using Ipopt, Optimization
import OptimizationMOI, OptimizationOptimJL
using DiskArrays: DiskArrays, eachchunk, arraysize_from_chunksize
using Statistics: mean
using StatsBase: mode
struct UndefinedChunks <: DiskArrays.ChunkType
  s::Int
end
Base.size(c::UndefinedChunks) = (1,)
Base.getindex(c::UndefinedChunks,_::Int) = UnitRange(0,-1)
DiskArrays.arraysize_from_chunksize(cs::UndefinedChunks)=cs.s

struct ExecutionPlan{N,P}
  input_chunkspecs
  output_chunkspecs
  sizes_raw::NTuple{N,Float64}
  windowsize::NTuple{N,Int}
  cost_min::Float64
  lr::P
end
function Base.show(io::IO,::MIME"text/plain",p::ExecutionPlan)
  comp = get(io, :compact, false)
  printinfo(io,p,extended=!comp)
end

function time_per_array(p::ExecutionPlan)
  input_times = time_per_array.(p.input_chunkspecs,(p.sizes_raw,))
  output_times = time_per_array.(p.output_chunkspecs,(p.sizes_raw,))
  (;input_times,output_times)
end
function time_per_chunk(p::ExecutionPlan)
  input_times = time_per_chunk.(p.input_chunkspecs,(p.sizes_raw,))
  output_times = time_per_chunk.(p.output_chunkspecs,(p.sizes_raw,))
  (;input_times,output_times)
end
function array_repeat_factor(p::ExecutionPlan) 
  input_times = array_repeat_factor.(p.input_chunkspecs,(p.windowsize,))
  output_times = array_repeat_factor.(p.output_chunkspecs,(p.windowsize,))
  (;input_times,output_times)
end
function access_per_chunk(p::ExecutionPlan)
  input_times = map(p.input_chunkspecs) do chunkspec
    mysize = mysub(chunkspec.lw,p.sizes_raw)
    access_per_chunk.(chunkspec.app_cs,mysize)
  end
  output_times = map(p.output_chunkspecs) do chunkspec
    mysize = mysub(chunkspec.lw,p.sizes_raw)
    map(chunkspec.app_cs,mysize) do acs,ms
      isnothing(acs) ? nothing : access_per_chunk(acs,ms)
    end
  end
  (;input_times,output_times)
end
function actual_access_per_chunk(p::ExecutionPlan)

  input_times = map(p.input_chunkspecs) do chunkspec
    mylr = mysub(chunkspec.lw,p.lr.members)
    actual_chunk_access.(chunkspec.cs,mylr,chunkspec.lw.windows.members)
  end
  output_times = map(p.output_chunkspecs) do chunkspec
    mylr = mysub(chunkspec.lw,p.lr.members)
    isnothing(chunkspec.cs) && return nothing
    actual_chunk_access.(chunkspec.cs,mylr,chunkspec.lw.windows.members)
  end
  (;input_times,output_times)
end
function actual_time_per_array(p::ExecutionPlan)
  input_times = map(p.input_chunkspecs) do chunkspec
    mylr = mysub(chunkspec.lw,p.lr.members)
    repfac = chunkspec.repfac
    prod(actual_chunk_access.(chunkspec.cs,mylr,chunkspec.lw.windows.members))*repfac*chunkspec.sr
  end
  output_times = map(p.output_chunkspecs) do chunkspec
    mylr = mysub(chunkspec.lw,p.lr.members)
    isnothing(chunkspec.cs) && return nothing
    prod(actual_chunk_access.(chunkspec.cs,mylr,chunkspec.lw.windows.members))*chunkspec.sr
  end
  (;input_times,output_times)
end
function actual_io_costs(p::ExecutionPlan)
  t = actual_time_per_array(p)
  s_in = mapreduce(+,zip(t.input_times,p.input_chunkspecs),init=0.0) do (ti,spec)
    s = arraysize_from_chunksize.(spec.cs)
    prod(s)*ti
  end
  s_out = mapreduce(+,zip(t.output_times,p.output_chunkspecs),init=0.0) do (ti,spec)
    s = arraysize_from_chunksize.(spec.cs)
    prod(s)*ti
  end
  s_in+s_out
end
function printinfo(io::IO,plan::ExecutionPlan;extended=true)
  n_chunks = length(plan.lr)
  sh_chunks = size(plan.lr)
  mean_windowsize = map(plan.lr.members) do w
    mode(length.(w))
  end
  println(io,"DiskArrayEngine ExecutionPlan")
  println(io,"=============================")
  println(io,"Processing in $n_chunks blocks of shape $sh_chunks")
  println(io,"With block sizes of approximately $mean_windowsize")
  println(io,"Sum of IO costs: $(actual_io_costs(plan))")
  extended || return nothing
  apc = access_per_chunk(plan)
  irt = time_per_chunk(plan)
  tpa = time_per_array(plan)
  arf = array_repeat_factor(plan)
  aapc = actual_access_per_chunk(plan)
  atpa = actual_time_per_array(plan)
  for (ii,ia) in enumerate(plan.input_chunkspecs)
    s = arraysize_from_chunksize.(ia.cs)
    println(io)
    println(io,"Input Array $ii of size $s")
    println(io,"----------------------------------")
    println(io,"Optim Access per chunk: $(apc.input_times[ii])")
    println(io,"Optim time per dim: $(irt.input_times[ii])")
    println(io,"With factor : $(arf.input_times[ii]) resulting in $(tpa.input_times[ii])")
    println(io,"Actual access per chunk: $(aapc.input_times[ii])")
    println(io,"Actual estimated readtime: $(atpa.input_times[ii]*prod(s))")
  end
  for (ii,ia) in enumerate(plan.output_chunkspecs)
    s = arraysize_from_chunksize.(ia.cs)
    println(io)
    println(io,"Output Array $ii of size $s")
    println(io,"---------------------------------")
    println(io,"Optim Access per chunk: $(apc.output_times[ii])")
    println(io,"Optim time per dim: $(irt.output_times[ii])")
    println(io,"With factor : $(arf.output_times[ii]) resulting in $(tpa.output_times[ii])")
    println(io,"Actual access per chunk: $(aapc.output_times[ii])")
    println(io,"Actual estimated writetime: $(atpa.output_times[ii]*prod(s))")
  end
end


access_per_chunk(cs,window) = cs/window
access_per_chunk(::Nothing,_) = 1.0
function integrated_readtime(_,cs::UndefinedChunks,singleread,window)
  cs.s/window*singleread    
end

"""
Given the loop windows of an input array estimate the apparent chunks along
an axis given the underlying chunks. 
"""
function apparent_chunksize(inar::InputArray)
  cs = eachchunk(inar.a)
  lwm = inar.lw.windows
  DiskArrays.GridChunks(apparent_chunksize.(cs.chunks,lwm.members))
end

function apparent_chunksize(cs, lw)
  lwcenters = mean.(lw)
  l = map(cs) do r
    searchsortedfirst(lwcenters,first(r))
    #length(searchsortedfirst(lwcenters,first(r)):searchsortedlast(lwcenters,last(r)))
  end
  push!(l,length(lwcenters)+1)
  l = diff(l)
  l = if all(<=(1),l)
    ones(Int, length(lw))
  else
    filter(!iszero,l)
  end
  sum(l) == length(lw) || error("Error in determining apparent chunk sizes")
  DiskArrays.chunktype_from_chunksizes(l)
end


function actual_chunk_access(cs,looprange,window)
  n_access = 0
  for lr in looprange 
    w1,w2 = mapreduce(((a,b),(c,d))->(min(a,c),max(b,d)),lr) do ii
      extrema(window[ii])
    end
    i = DiskArrays.findchunk(cs,w1:w2)
    n_access = n_access+length(i)
  end
  n_access/length(cs)
end

actual_chunk_access(cs::UndefinedChunks,looprange,window) = 1.0


function integrated_readtime(app_cs,cs,singleread,window) 
  acp = access_per_chunk(app_cs,window)
  readtime = if acp < 1.0
    #length(cs)*singleread*(1-0.5*window/maximum(last(cs)))
    length(cs)*singleread*(0.999 + 0.001*acp)
  else
    length(cs)*acp*singleread
  end
  readtime
end

function get_chunkspec(outspec,ot)
  cs = outspec.chunks
  avgs = avg_step.(outspec.lw.windows.members)
  si = map(m->last(last(m))-first(first(m))+1,outspec.lw.windows.members)
  if cs isa GridChunks
    cs = cs.chunks
  elseif cs === nothing
    cs = map(_->nothing,si)
  end
  cs = map(cs,si) do csnow,s
    if csnow === nothing
      return UndefinedChunks(s)
    end
    if csnow isa Integer
      DiskArrays.RegularChunks(csnow,0,s)
    else
      csnow
    end
  end
  app_cs = map(get_app_cs,cs,avgs) 
  sr = estimate_singleread(outspec)
  lw = outspec.lw
  windowfac = avgs
  repfac = 1.0
  windowoffset = max_size.(outspec.lw.windows.members)
  elsize = sizeof(Base.nonmissingtype(ot))
  (;cs,app_cs,sr,lw,elsize,windowfac,windowoffset,repfac)
end

get_app_cs(cs,avgs) = ceil(Int,DiskArrays.approx_chunksize(cs) / avgs)
get_app_cs(::UndefinedChunks,_) = nothing
function get_chunkspec(ia::InputArray,totsize)
  cs = DiskArrays.eachchunk(ia.a).chunks
  avgs = avg_step.(ia.lw.windows.members)
  app_cs = get_app_cs.(cs,avgs)
  sr = estimate_singleread(ia)
  lw = ia.lw
  windowfac = avgs
  repfac = array_repeat_factor(lw,totsize)
  windowoffset = max_size.(ia.lw.windows.members)
  elsize = DiskArrays.element_size(ia.a)
  (;cs,app_cs,sr,lw,elsize,windowfac,windowoffset,repfac)
end
function array_repeat_factor(lw,totsize)
  mytot = mysub(lw,totsize)
  prod(totsize)/prod(mytot)
end
function time_per_array(spec,window)
  repfac = spec.repfac
  prod(time_per_chunk(spec,window))*repfac
end
function time_per_chunk(spec,window)
  mywindow = mysub(spec.lw,window)
  integrated_readtime.(spec.app_cs,spec.cs,spec.sr,mywindow)
end


function bufsize_per_array(spec,window)
  wsizes = mysub(spec.lw,window)
  prod((spec.windowoffset .+ spec.windowfac .* (wsizes .- 1)))*spec.elsize
end

compute_bufsize(window,chunkspec...) = sum(bufsize_per_array.(chunkspec,(window,)))
function compute_time(window,chunkspec) 
  sum(time_per_array.(chunkspec,(window,)))
end
all_constraints(window,chunkspec) = (compute_bufsize(window,chunkspec...),window...)
all_constraints!(res,window,chunkspec) = res.=all_constraints(window,chunkspec)

avg_step(lw) = avg_step(lw,get_ordering(lw),get_overlap(lw))
avg_step(lw,::Union{Increasing,Decreasing},::Any) = length(lw) > 1 ? mean(diff(first.(lw))) : 1.0
avg_step(lw,::Any,::Any) = error("Not implemented")
max_size(lw) = maximum(length,lw)

estimate_singleread(ia::InputArray)= ismem(ia) ? 1e-16 : 1.0
estimate_singleread(ia) = ia.ismem ? 1e-16 : 3.0

function optimize_loopranges(op::GMDWop,max_cache;tol_low=0.2,tol_high = 0.05,max_order=2,x0 = nothing, force_regular=false)
  lb = [0.0,map(_->1.0,op.windowsize)...]
  ub = [max_cache,op.windowsize...]
  x0 = x0 === nothing ? [2.0 for _ in op.windowsize] : x0
  totsize = op.windowsize
  input_chunkspecs = get_chunkspec.(op.inars,(totsize,))
  output_chunkspecs = get_chunkspec.(op.outspecs,op.f.outtype)
  chunkspecs = (input_chunkspecs..., output_chunkspecs...)
  optprob = OptimizationFunction(compute_time, Optimization.AutoForwardDiff(), cons = all_constraints!)
  prob = OptimizationProblem(optprob, x0, chunkspecs, lcons = lb, ucons = ub)
  sol = solve(prob, OptimizationOptimJL.IPNewton())
  @debug "Optimized Loop sizes: ", sol.u
  lr = adjust_loopranges(op,sol.u;tol_low,tol_high,max_order,force_regular)
  ExecutionPlan(input_chunkspecs, output_chunkspecs,(sol.u...,),totsize,sol.objective,lr)
end

using OrderedCollections, Primes

function kgv(i...)
  f = LittleDict.(factor.(i))
  prod((i)->first(i)^last(i),merge(max,f...))
end
kgv(i)=i


function is_possible_candidate(cand,smax,optires,reltol_low,reltol_high)
  reltol = cand > optires ? reltol_high : reltol_low
  cand<=smax && abs(cand-optires)/optires<reltol
end

function find_adjust_candidates(optires,smax,intsizes;reltol_low=0.2,reltol_high=0.05,max_order=2)
  smallest_common = kgv(intsizes...)
  if optires > smallest_common
    for ord in 1:max_order 
      rr = round(Int,optires/smallest_common*ord)
      cand = smallest_common * rr//ord
      is_possible_candidate(cand,smax,optires,reltol_low,reltol_high) && return cand
    end
    #Did not find a better candidate, try rounding
  elseif smallest_common < smax
    for ord in 1:max_order 
      rr = round(Int,smallest_common/optires*ord)
      cand = smallest_common * ord // rr
      is_possible_candidate(cand,smax,optires,reltol_low,reltol_high) && return cand
    end
  end
  if length(intsizes) > 1
    #Simply try with less input arrays, to at least align a few of them, this could be further optimized
    return find_adjust_candidates(optires,smax,Base.front(intsizes);reltol_low, reltol_high,max_order)
  end
  cand = round(Int,optires)//1
  is_possible_candidate(cand,smax,optires,reltol_low,reltol_high) && return cand
  return floor(Int,optires)//1
end

function generate_LoopRange(r_adj::Rational,apparent_chunks::ChunkType;tres=3)
  splitsize = ceil(Int,r_adj)
  all_ends = last.(apparent_chunks)
  firstend = findlast(<=(splitsize),all_ends)
  res,inow = if firstend === nothing
    Int[], 1
  else
    Int[all_ends[firstend]], all_ends[firstend]+1
  end
  while inow <= last(last(apparent_chunks))
    scand = inow+splitsize-1
    iallends = searchsortedlast(all_ends,scand)
    if iallends > 0 
      if iallends > length(all_ends)
        push!(res,last(all_ends)-inow+1)
        inow = last(all_ends)+1
      elseif abs(all_ends[iallends]-scand) < tres
        push!(res,all_ends[iallends]-inow+1)
        inow = all_ends[iallends]+1
      else
        push!(res,min(splitsize,last(all_ends)-inow+1))
        inow = inow+splitsize
      end
    else
      push!(res,splitsize)
      inow = inow+splitsize
    end
  end
  res
end



function adjust_loopranges(optotal,approx_opti;tol_low=0.2,tol_high = 0.05,max_order=2, force_regular=false)
  inars = filter(!ismem,optotal.inars)
  app_cs = apparent_chunksize.(inars)
  r = map(approx_opti,1:length(approx_opti),optotal.windowsize) do sol,iopt,si
    inaxchunks = ()
    for ia in 1:length(inars)
      li = getloopinds(inars[ia].lw)
      if iopt in li
        ili = findfirst(==(iopt),li)
        inaxchunks = (inaxchunks...,(app_cs[ia].chunks[ili]))
      end
    end
    if !isempty(inaxchunks)
      insizes = DiskArrays.approx_chunksize.(inaxchunks)
      cands = find_adjust_candidates(sol,si,insizes;reltol_low=tol_low,reltol_high=tol_high,max_order)
      cands, first(inaxchunks)
    else
      rsol = clamp(round(Int,sol),1,si)
      rsol//1, RegularChunks(rsol,0,si)
    end
  end
  
  adj_cands = first.(r)
  adj_chunks = last.(r)
  

  @debug "Adjust candidates: ", adj_cands
  lr = if force_regular
    DiskArrays.RegularChunks.(round.(Int,adj_cands),0,optotal.windowsize)
  else
    lr = generate_LoopRange.(adj_cands,adj_chunks,tres=3)
    DiskArrays.chunktype_from_chunksizes.(fix_output_overlap(optotal.outspecs,lr))
  end
  foreach(lr,optotal.windowsize) do l,s
    @assert first(first(l))>=1
    @assert last(last(l))<=s
  end
  ProductArray((lr...,))
end

"""
If one of the outputs is a reduction it is important not to have overlapping
loop ranges for a reduction group. This will try to correct loopranges to avoid
the problems mentioned above.
"""
function fix_output_overlap(outspecs,lrbreaks)
  for outspec in outspecs
    mylr = mysub(outspec.lw,lrbreaks)
    newbreaks = map(mylr,outspec.lw.windows.members) do breaks,window
      if get_overlap(window) isa Repeating
        r = collect(DiskArrays.chunktype_from_chunksizes(breaks))
        for i in 1:length(r)-1
          r1,r2 = r[i],r[i+1]
          split_orig = last(r1)
          moveleft = length(r1)>length(r2)
          isplit = split_orig
          r1_array, r2_array = window[first(r1)]:window[last(r1)],window[first(r2)]:window[last(r2)]
          while !isempty(intersect(r1_array,r2_array)) && r1 != r2
            isplit = moveleft ? (isplit - 1) : (isplit + 1)
            r1 = first(r1):isplit
            r2 = (isplit+1):last(r2)
            r1_array, r2_array = window[first(r1)]:window[last(r1)],window[first(r2)]:window[last(r2)]
          end
          r[i]=r1
          r[i+1]=r2
        end
        length.(r)
      else
        breaks
      end
    end
    for (lr,b) in zip(mylr,newbreaks)
      lr.=b
    end
  end
  lrbreaks
end


function output_chunks(outspec,lr)
  mylr=mysub(outspec.lw,lr.members)
  map(mylr,outspec.lw.windows.members) do llr,wi
    ww = map(llr) do l
      wnow = wi[l]
      first(first(wnow)):last(last(wnow))
    end
    DiskArrays.chunktype_from_chunksizes(length.(sort(unique(ww),lt=rangelt)))
  end
end

function output_chunks(p::ExecutionPlan)
output_chunks.(p.output_chunkspecs,(p.lr,))
end