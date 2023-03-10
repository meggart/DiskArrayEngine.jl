using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Interpolations
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: MWOp, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, results_as_diskarrays, create_userfunction, steps_per_chunk, apparent_chunksize,
  find_adjust_candidates, generate_LoopRange, get_loopsplitter, split_loopranges_threads, merge_loopranges_threads, MovingWindow, RegularWindows
using StatsBase: rle
using CFTime: timedecode
using Dates
a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

t = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/time/", fill_as_missing=true)

function getinterpinds(oldvals::AbstractRange, newvals::AbstractRange)
  (newvals.-first(oldvals))./step(oldvals).+1
end
function getinterpinds(r1,r2)
  rev = issorted(r1) ? false : issorted(r1,rev=true) ? true : error("Axis values are not sorted")
  map(r2) do ir
    ii = searchsortedfirst(r1,ir,rev=rev)
    ii1 = max(min(ii-1,length(r1)),1)
    ii2 = max(min(ii,length(r1)),1)
    ind = if ii1 == ii2
      Float64(ii1)
    else
      ii1+(ir-r1[ii1])/(r1[ii2]-r1[ii1])
    end
    ind
  end
end


function getallsteps(xcoarse,xfine)
  interpinds = getinterpinds(xcoarse, xfine)
  resout = UnitRange{Int}[]
  icur = 1
  while icur <= length(interpinds)
    i1 = floor(interpinds[icur])
    inext = searchsortedlast(interpinds,i1+1)
    push!(resout,icur:inext)
    icur = inext+1
  end

  resin = DiskArrayEngine.MovingWindow(floor(Int,first(interpinds)),1,2,length(resout))
  interpinds, resin, resout
end


lons = range(-179.875,179.875,length=1440)
lats = range(89.875,-89.875,length=720)
ts = t[:]

newlons = range(-180.0,180.0,length=2881)
newlats = range(90,-90,length=1441)
ts
newts = 366.0:1.0:14615

conv = (1=>(lons,newlons),2=>(lats,newlats))
xcoarse = lats
xfine = newlats
interpinds = getinterpinds(xcoarse, xfine)
resout = UnitRange{Int}[]
icur = 1
#while icur <= length(interpinds)
  i1 = floor(interpinds[icur])
  inext = searchsortedlast(interpinds,i1+1)
  push!(resout,icur:inext)
  icur = inext+1
#end

resin = DiskArrayEngine.MovingWindow(floor(Int,first(interpinds)),1,2,length(resout))
interpinds, resin, resout


getallsteps(lats,newlats)



allinfo = [k=>getallsteps(v...) for (k,v) in conv]

function interpolate_block!(xout, data, nodes...; dims,threaded=false)
  innodes = axes(data)
  isinterp = ntuple(in(dims),ndims(data))
  modes = map(isinterp) do m
    m ? BSpline(Linear()) : NoInterp()
  end
  outnodes = innodes
  for d in dims
    n,nodes = first(nodes),Base.tail(nodes)
    outnodes = Base.setindex(outnodes,n,d)
  end
  itp = extrapolate(interpolate(data,modes), Flat())
  fill_nodes!(itp,xout,outnodes)
end
@noinline function fill_nodes!(itp,xout,outnodes)
  pa = ProductArray(outnodes)
  broadcast!(Base.splat(itp),xout,pa)
end


f = create_userfunction(interpolate_block!,Union{Float32,Missing},is_blockfunction=true,is_mutating=true,dims=3)


rp = ProductArray((1:size(a,1),1:size(a,2),stepin))
inar1 = InputArray(a,LoopWindows(rp,Val((1,2,3))))

rp2 = ProductArray((stepout,))
inar2 = InputArray(interpinds,LoopWindows(rp2,Val((3,))))

inars = (inar1,inar2)

rpout = ProductArray((1:size(a,1),1:size(a,2),stepout))
outwindows = ((lw = LoopWindows(rpout,Val((1,2,3))), chunks = eachchunk(outar).chunks, ismem=true),)

optotal = GMDWop(inars, outwindows, f)
r, = results_as_diskarrays(optotal)


rr = r[1000,300,:];
a[1000,300,:]


outar = zcreate(Union{Float32,Missing},size(a)[1:2]...,length(newts),
  path=tempname(),chunks = (90,90,500),fill_value=typemax(Float32))

function run_op(op,outars;max_cache=5e8,threaded=true)
  lr = DiskArrayEngine.optimize_loopranges(op,max_cache,tol_low=0.2,tol_high=0.05,max_order=2)
  DiskArrayEngine.run_loop(optotal,lr,outars,threaded=true)
end

lr = DiskArrayEngine.optimize_loopranges(optotal,1e8,tol_low=0.2,tol_high=0.05,max_order=2)

@time run_op(optotal, (outar,),threaded=false,max_cache=5e8)

using Plots

length(rr)

p = plot(ts,a[1000,300,:])
plot!(newts,rr)

DiskArrayEngine.getoutspec(r).windows

r.op.windowsize

astepin

#Aggregation to daily
data = a[subset...]
datao = OffsetArray(data,subset)

xout = OffsetArray(zeros(11,11,801),subset[1],subset[2],newts)

interpolate_block!(xout,datao,inds,dims=3)

(1000, 300, 13:14) .- (1,2,3)

using Plots

p = plot(newts,rr)
plot!(p,ts,a[1000,300,:])



using Zarr, Blosc

data = rand(10000)

srcbuf = Blosc.compress(data)

s1,s2,bs = Blosc.sizes(srcbuf)

Int(bs)