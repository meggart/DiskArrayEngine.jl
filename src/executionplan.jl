using Ipopt, Optimization, OptimizationMOI, OptimizationOptimJL
struct UndefinedChunks 
    s::Int
end


access_per_chunk(cs,window) = cs/window
function integrated_readtime(_,cs::UndefinedChunks,singleread,window)
    cs.s/window*singleread    
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
            csnow = DiskArrays.RegularChunks(csnow,0,s)
        end
    end
    app_cs = map(cs,avgs) do csnow,avgsnow
        if csnow isa UndefinedChunks 
            nothing
        else
            ceil(Int,DiskArrays.approx_chunksize.(csnow) / avgsnow)
        end
    end
    sr = estimate_singleread(outspec)
    lw = outspec.lw
    windowfac = prod(avgs)
    elsize = sizeof(Base.nonmissingtype(ot))
    (;cs,app_cs,sr,lw,elsize,windowfac)
end

function get_chunkspec(ia::InputArray)
    cs = mysub(ia.lw,DiskArrays.eachchunk(ia.a).chunks)
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

estimate_singleread(ia::InputArray)=1.0
estimate_singleread(ia) = ia.ismem ? 1e-8 : 3.0  

function optimize_loopranges(op::GMDWop,max_cache)
  lb = [0.0,map(_->1.0,op.windowsize)...]
  ub = [max_cache,op.windowsize...]
  x0 = [2.0 for _ in op.windowsize]
  totsize = op.windowsize
  chunkspecs = (totsize,get_chunkspec.(op.inars)..., get_chunkspec.(op.outspecs,op.f.outtype)...)
  optprob = OptimizationFunction(compute_time, Optimization.AutoForwardDiff(), cons = all_constraints!)
  prob = OptimizationProblem(optprob, x0, chunkspecs, lcons = lb, ucons = ub)
  @show compute_time(x0,chunkspecs)
  @show all_constraints(x0,chunkspecs)
  sol = solve(prob, IPNewton())
end