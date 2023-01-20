using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Interpolations
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: ProcessingSteps, MWOp, subset_step_to_chunks, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  NonMutatingFunction, create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, MutatingBlockFunction
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


lons = range(-179.875,179.875,length=1440)
lats = range(89.875,-89.875,length=720)
ts = t[:]
ts
newts = 2376:3176
inds = OffsetArray(getinterpinds(ts,newts),newts)
subset = (200:210,200:210,300:400)




function interpolate_block!(xout, data, nodes...; dims)
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
  #@show itp
  #@show eachindex(xout)
  pa = ProductArray(outnodes)
  broadcast!(Base.splat(itp),xout,pa)
end

mainfunc = MutatingBlockFunction(interpolate_block)
init = zero(Float32)
filters = nothing
fin(x) = identity
outtypes = (Union{Float32,Missing},)
args = ()
kwargs = (;dims=3)
f = UserOp(mainfunc,nothing,init,filters,identity,outtypes,args,kwargs)

struct MovingWindow <: AbstractVector{UnitRange{Int}}
  first::Int
  steps::Int
  width::Int
  n::Int
end
Base.size(m::MovingWindow) = (m.n,)
Base.getindex(m::MovingWindow,i) = (m.first+(i-1)*m.steps):(m.first+(i-1)*m.steps+m.width-1)

m = MovingWindow(1,1,2,length(ts)-1)


m
stepveclat = ProcessingSteps(0,1:size(a,2))
stepveclon = ProcessingSteps(0,1:size(a,1))
#Chunks over the loop indices


#First example: Latitudinal mean of monthly average
o1 = MWOp(eachchunk(a).chunks[1])
o2 = MWOp(eachchunk(a).chunks[2])
o3 = MWOp(eachchunk(a).chunks[3],steps=stepvectime)

ops = (o1,o2,o3)

optotal = GMDWop(inars, outwindows, f)
r, = results_as_diskarrays(optotal)


#Aggregation to daily
data = a[subset...]
datao = OffsetArray(data,subset)

xout = OffsetArray(zeros(11,11,801),subset[1],subset[2],newts)

interpolate_block!(xout,datao,inds,dims=3)

using Plots

p = plot(xout[200,200,:])
plot!(p,(297:397).*8,parent(datao[200,200,:]))

datao[200,200,:]