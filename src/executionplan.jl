using Ipopt, Optimization
import OptimizationMOI, OptimizationOptimJL
using DiskArrays: eachchunk
using Statistics: mean
struct UndefinedChunks 
    s::Int
end


access_per_chunk(cs,window) = cs/window
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
    length(searchsortedfirst(lwcenters,first(r)):searchsortedlast(lwcenters,last(r)))
  end
  l = if all(iszero,l)
    ones(Int, length(lw))
  else
    filter(!iszero,l)
  end
  sum(l) == length(lw) || error("Error in determining apparent chunk sizes")
  DiskArrays.chunktype_from_chunksizes(l)
end


function integrated_readtime(app_cs,cs,singleread,window) 
  acp = access_per_chunk(app_cs,window)
  if acp < 1.0
    length(cs)*singleread*(1-0.5*window/maximum(last(cs)))
  else
    length(cs)*acp*singleread
  end
end

function get_chunkspec(outspec,ot)
    cs = outspec.chunks
    avgs = avg_step.(outspec.lw.windows.members)
    si = map(m->last(last(m))-first(first(m))+1,outspec.lw.windows.members)
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
    app_cs = map(cs,avgs) do csnow,avgsnow
        if csnow isa UndefinedChunks 
            nothing
        else
            ceil(Int,DiskArrays.approx_chunksize(csnow) / avgsnow)
        end
    end
    sr = estimate_singleread(outspec)
    lw = outspec.lw
    windowfac = prod(avgs)
    elsize = sizeof(Base.nonmissingtype(ot))
    (;cs,app_cs,sr,lw,elsize,windowfac)
end

function get_chunkspec(ia::InputArray)
    cs = DiskArrays.eachchunk(ia.a).chunks
    avgs = avg_step.(ia.lw.windows.members)
    app_cs = ceil.(Int,DiskArrays.approx_chunksize.(cs) ./ avgs)
    sr = estimate_singleread(ia)
    lw = ia.lw
    windowfac = prod(avgs)
    elsize = DiskArrays.element_size(ia.a)
    (;cs,app_cs,sr,lw,elsize,windowfac)
end
function time_per_array(spec,window,totsize)
    mytot = mysub(spec.lw,totsize)
    repfac = prod(totsize)/prod(mytot)
    mywindow = mysub(spec.lw,window)
    prod(integrated_readtime.(spec.app_cs,spec.cs,spec.sr,mywindow))*repfac
end
function bufsize_per_array(spec,window)
    prod(mysub(spec.lw,window))*spec.elsize*spec.windowfac
end

compute_bufsize(window,_,chunkspec...) = sum(bufsize_per_array.(chunkspec,(window,)))
function compute_time(window,chunkspec) 
    totsize = first(chunkspec)
    chunkspec = Base.tail(chunkspec)
    sum(time_per_array.(chunkspec,(window,),(totsize,)))
end
all_constraints(window,chunkspec) = (compute_bufsize(window,chunkspec...),window...)
all_constraints!(res,window,chunkspec) = res.=all_constraints(window,chunkspec)
  
avg_step(lw) = (last(last(lw))-first(first(lw))+1)/length(lw)

estimate_singleread(ia::InputArray)= ismem(ia) ? 1e-8 : 1.0
estimate_singleread(ia) = ia.ismem ? 1e-8 : 3.0  

function optimize_loopranges(op::GMDWop,max_cache;tol_low=0.2,tol_high = 0.05,max_order=2)
  lb = [0.0,map(_->1.0,op.windowsize)...]
  ub = [max_cache,op.windowsize...]
  x0 = [2.0 for _ in op.windowsize]
  totsize = op.windowsize
  chunkspecs = (totsize,get_chunkspec.(op.inars)..., get_chunkspec.(op.outspecs,op.f.outtype)...)
  optprob = OptimizationFunction(compute_time, Optimization.AutoForwardDiff(), cons = all_constraints!)
  prob = OptimizationProblem(optprob, x0, chunkspecs, lcons = lb, ucons = ub)
  sol = solve(prob, OptimizationOptimJL.IPNewton())
  @debug "Optimized Loop sizes: ", sol.u
  adjust_loopranges(op,sol;tol_low,tol_high,max_order)
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
      return find_adjust_candidates(optires,smax,Base.tail(intsizes);reltol_low, reltol_high,max_order)
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
    DiskArrays.chunktype_from_chunksizes(res)
  end

  

  function adjust_loopranges(optotal,approx_opti;tol_low=0.2,tol_high = 0.05,max_order=2)
    inars = filter(!ismem,optotal.inars)
    app_cs = apparent_chunksize.(inars)
    r = map(approx_opti.u,1:length(approx_opti.u),optotal.windowsize) do sol,iopt,si
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
    lr = generate_LoopRange.(adj_cands,adj_chunks,tres=3)
    foreach(lr,optotal.windowsize) do l,s
        @assert first(first(l))>=1
        @assert last(last(l))<=s
    end
    ProductArray((lr...,))
  end
  
