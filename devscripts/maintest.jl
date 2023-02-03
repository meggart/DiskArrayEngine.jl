using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: MWOp, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
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
outwindows = ((lw=LoopWindows(outrp,Val((2,3))),chunks=(nothing, nothing),ismem=false),)

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

DiskArrays.approx_chunksize.(eachchunk(a).chunks)

using OrderedCollections

function kgv(i...)
  f = LittleDict.(factor.(i))
  prod((i)->first(i)^last(i),merge(max,f...))
end
kgv(i)=i
kgv(90,60,70)

optires = 1333
intsizes = 1000
smax=1_000_000
find_adjust_candidates

function find_adjust_candidates(optires,smax,intsizes;reltol=0.05,max_order=2)
  smallest_common = kgv(intsizes...)
  if optires > smallest_common
    for ord in 1:max_order 
      rr = round(Int,optires/smallest_common*ord)
      if rr<smax && abs(rr-optires/smallest_common*ord)/(optires/smallest_common*ord)<reltol
        return smallest_common * rr//ord
      end
    end
    #Did not find a better candidate, try rounding
    if abs(round(Int,optires)-optires)/optires < reltol
      return round(Int,optires)
    else
      return floor(optires)
    end
  else

  end
end

optires = 1333
intsizes = (1000,)
smax=1_000_000
find_adjust_candidates(optires,smax,intsizes,max_order=3)

DiskArrayEngine.optimize_loopranges(optotal,5e8)

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
csvec = [10:90;95:5:200]

readtime = [@elapsed extract_slice(a,cs) for cs in csvec]
p = plot(csvec,readtime,log="x")
ticvec = [18,20,30,36,45,60,90,120,135,150,180]
xticks!(p,ticvec)
vline!(p,ticvec)

using DiskArrays: approx_chunksize
using DiskArrayEngine: RegularWindows

singleread = median([@elapsed a[first(eachchunk(a))...] for _ in 1:10])

#p = plot(csvec,integrated_readtime.((eachchunk(a).chunks[1],),singleread,csvec))
#plot!(p,csvec,readtime)


#2 example arrays
p1 = tempname()
p2 = tempname()
a1 = zcreate(Float32,10000,10000,path = p1, chunks = (10000,1),fill_value=2.0,fill_as_missing=false)
a2 = zcreate(Float32,10000,10000,path = p2, chunks = (1,10000),fill_value=5.0,fill_as_missing=false)



eltype(r)

size(r)

rp = ProductArray((1:10000,DiskArrayEngine.RegularWindows(1,10000,step=3)))

# rangeproduct[3]
inars = (InputArray(a1,LoopWindows(rp,Val((1,2)))),InputArray(a2,LoopWindows(rp,Val((1,2)))))

outrp = ProductArray(())
outwindows = ((lw=LoopWindows(outrp,Val(())),chunks=(),ismem=false),)

f = create_userfunction(
    +,
    Float64,
    red = +, 
    init = 0.0,   
)

optotal = GMDWop(inars, outwindows, f)

DiskArrayEngine.optimize_loopranges(optotal,1e8)

compute_time(window,arraychunkspec)
compute_bufsize(window,arraychunkspec)
all_constraints(window,arraychunkspec)

using Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
using ForwardDiff, ModelingToolkit
window = [1000,1000]
loopsize = (10000,10000)







using BenchmarkTools, Skipper

nanmin(x,y) = isnan(x) ? y : isnan(y) ? x : min(x,y)
nanmax(x,y) = isnan(x) ? y : isnan(y) ? x : min(x,y)

nanminimum(a;kwargs...) = reduce(nanmin,a;init=NaN,kwargs...)
nanminimum(f,a;kwargs...) = mapreduce(f,nanmin,a;init=NaN,kwargs...)

x = rand(10000,10);
nanminimum(x,dims=1)

skip(isnan,x) |> typeof

x[rand(1:length(x),1000)] .= NaN;
@benchmark reduce(nanmin,$x,dims=2,init=Inf)


@benchmark minimum.(skip.(isnan, eachslice($x, dims=2)))