using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: ProcessingSteps, MWOp, subset_step_to_chunks, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, results_as_diskarrays, create_userfunction, steps_per_chunk
using StatsBase: rle
using CFTime: timedecode
using Dates
using OnlineStats
a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

t = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/time/", fill_as_missing=true)
t.attrs["units"]
tvec = timedecode(t[:],t.attrs["units"])
years, nts = rle(yearmonth.(tvec))
cums = [0;cumsum(nts)]

stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(nts)]


stepveclat = 1:size(a,2)
stepveclon = 1:size(a,1)

rp = ProductArray((stepveclon,stepveclat,stepvectime))

# rangeproduct[3]
inars = (InputArray(a,LoopWindows(rp,Val((1,2,3)))),)


outrp = ProductArray((stepveclat,1:length(stepvectime)))
outwindows = (LoopWindows(outrp,Val((2,3))),)

function myfunc(x)
  all(ismissing,x) ? (0,zero(eltype(x))) : (1,mean(skipmissing(x)))
end

function reducefunc((n1,s1),(n2,s2))
  (n1+n2,s1+s2)
end
init = ()->(0,zero(Float64))
filters = (NoFilter(),)
fin(x) = last(x)/first(x)
outtypes = (Union{Float32,Missing},)
args = ()
kwargs = (;)
f = create_userfunction(
  myfunc,
  Union{Float32,Missing},
  red = reducefunc, 
  init = init, 
  finalize=fin,
  buftype = Tuple{Int,Union{Float32,Missing}},  
)

f.buftype

optotal = GMDWop(inars, outwindows, f)


r, = results_as_diskarrays(optotal)
rsub = r[300:310,200:210]





function fit_online!(xout,x)
  if !all(ismissing,x) 
    fit!(xout[],mean(skipmissing(x)))
  end
end
init = ()->OnlineStats.Mean()
filters = (NoFilter(),)
fin_onine(x) = nobs(x) == 0 ? missing : OnlineStats.value(x)
f = create_userfunction(
    fit_online!,
    Float64,
    is_mutating = true,
    red = OnlineStats.merge!, 
    init = init, 
    finalize=fin_onine,
    buftype = Mean,  
)


optotal = GMDWop(inars, outwindows, f)


r, = results_as_diskarrays(optotal)
rsub = r[300:310,200:210]

eachchunk(r)





loopranges = ProductArray((eachchunk(a).chunks[1:2]...,DiskArrays.RegularChunks(120,0,480)))
b = zeros(Union{Float32,Missing},size(a,2),length(stepvectime));
outars = (InputArray(b,outwindows[1]),)
DiskArrayEngine.run_loop(optotal,loopranges,outars)


using Plots
heatmap(b)


#Test for time to extract series of longitudes
cs = 100
function extract_slice(a,cs)
  r = zeros(Union{Missing,Float32},1440)
  for i in 1:cs:1440
    r[i:min(1440,i+cs-1)] .= a[i:min(1440,i+cs-1)]
  end
  r
end
csvec = 10:90

readtime = [@elapsed extract_slice(a,cs) for cs in csvec]
p = plot(csvec,readtime,log="x")
ticvec = [18,20,30,36,45,60,90,120,135,150,180]
xticks!(p,ticvec)
vline!(p,ticvec)

using DiskArrays: approx_chunksize

singleread = median([@elapsed a[first(eachchunk(a))...] for _ in 1:10])
access_per_chunk(cs,window) = cs/window
function integrated_readtime(app_cs,cs,singleread,window) 
  acp = access_per_chunk(app_cs,window)
  if acp < 1.0
    length(cs)*singleread*(1-0.5*window/maximum(last(cs)))
  else
    length(cs)*acp*singleread
  end
end
#p = plot(csvec,integrated_readtime.((eachchunk(a).chunks[1],),singleread,csvec))
#plot!(p,csvec,readtime)


#2 example arrays
p1 = tempname()
p2 = tempname()
a1 = zcreate(Float32,10000,10000,path = p1, chunks = (10000,1),fill_value=2.0,fill_as_missing=false)
a2 = zcreate(Float32,10000,10000,path = p2, chunks = (1,10000),fill_value=5.0,fill_as_missing=false)


rp = ProductArray((1:10000,1:10000))

# rangeproduct[3]
inars = (InputArray(a1,LoopWindows(rp,Val((1,2)))),InputArray(a2,LoopWindows(rp,Val((1,2)))))



singleread = (1.0,1.0)
estimate_singleread(ia)=1.0
arraychunkspec = collect(map(inars) do ia
  cs = mysub(ia.lw,eachchunk(ia.a).chunks)
  app_cs = DiskArrays.approx_chunksize.(cs)
  sr = estimate_singleread(ia)
  lw = ia.lw
  elsize = sizeof(eltype(ia.a))
  (;cs,app_cs,sr,lw,elsize)
end)
function time_per_array(spec,window)
  prod(integrated_readtime.(spec.app_cs,spec.cs,spec.sr,window))
end
function bufsize_per_array(spec,window)
  prod(mysub(spec.lw,window))*spec.elsize
end

compute_bufsize(window,chunkspec) = sum(bufsize_per_array.(chunkspec,(window,)))
window = [1000,1000]
compute_time(window,chunkspec) = sum(time_per_array.(chunkspec,(window,)))
all_constraints(window,chunkspec) = (compute_bufsize(window,chunkspec),window...)

compute_time(window,arraychunkspec)
compute_bufsize(window,arraychunkspec)
all_constraints(window,arraychunkspec)
all_constraints!(res,window,chunkspec) = res.=all_constraints(window,chunkspec)



using Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
using ForwardDiff, ModelingToolkit

loopsize = (10000,10000)
max_cache=1e7
lb = [0.0,map(_->1.0,window)...]
ub = [max_cache,loopsize...]
x0 = [2.0,2.0]
optprob = OptimizationFunction(compute_time, Optimization.AutoForwardDiff(), cons = all_constraints!)
prob = OptimizationProblem(optprob, x0, arraychunkspec, lcons = lb, ucons = ub)
sol = solve(prob, IPNewton())