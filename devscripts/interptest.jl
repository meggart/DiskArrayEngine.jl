using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Interpolations
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: ProcessingSteps, MWOp, subset_step_to_chunks, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, MovingWindow, create_userfunction, results_as_diskarrays
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




function getinterpsteps(xcoarse,xfine)
  ifine = 1
  resout = UnitRange{Int}[]
  resin = UnitRange{Int}[]
  xcs = findfirst(>=(first(xfine)),xcoarse)
  for xc in xcs:length(xcoarse)-1
    inew = findnext(>=(xcoarse[xc+1]),xfine,ifine)
    if inew===nothing
      push!(resout,ifine:length(xfine))
      break
    end
    push!(resout,ifine:inew)
    ifine = inew+1
  end
  resin = MovingWindow(xcs,1,2,length(resout))
  resin,resout
end


lons = range(-179.875,179.875,length=1440)
lats = range(89.875,-89.875,length=720)
ts = t[:]
ts
newts = 100.0:14612

inds

stepin, stepout = getinterpsteps(ts,newts)





function interpolate_block!(xout, data, nodes...; dims)
  innodes = axes(data)
  @show axes(xout)
  @show length(nodes[1])
  isinterp = ntuple(in(dims),ndims(data))
  modes = map(isinterp) do m
    m ? BSpline(Linear()) : NoInterp()
  end
  outnodes = innodes
  for d in dims
    n,nodes = first(nodes),Base.tail(nodes)
    outnodes = Base.setindex(outnodes,n,d)
  end
  # @show innodes
  # @show outnodes[1]
  # @show outnodes[2]
  # @show first(outnodes[3]),last(outnodes[3])
  # @show data
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

inds = getinterpinds(ts,newts)
rp2 = ProductArray((stepout,))
inar2 = InputArray(inds,LoopWindows(rp2,Val((3,))))

inars = (inar1,inar2)

rpout = ProductArray((1:size(a,1),1:size(a,2),stepout))
outwindows = (LoopWindows(rpout,Val((1,2,3))),)

optotal = GMDWop(inars, outwindows, f)
r, = results_as_diskarrays(optotal)


rr = r[1000,300,:];

plot(rr)

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