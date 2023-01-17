using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics

using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: ProcessingSteps, MWOp, subset_step_to_chunks, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  NonMutatingFunction, create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop
using StatsBase: rle
using CFTime: timedecode
using Dates
a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

t = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/time/", fill_as_missing=true)
t.attrs["units"]
tvec = timedecode(t[:],t.attrs["units"])
years, nts = rle(yearmonth.(tvec))
cums = [0;cumsum(nts)]

stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(nts)]


stepveclat = ProcessingSteps(0,1:size(a,2))
stepveclon = ProcessingSteps(0,1:size(a,1))

#Chunks over the loop indices


#First example: Latitudinal mean of monthly average
o1 = MWOp(eachchunk(a).chunks[1])
o2 = MWOp(eachchunk(a).chunks[2])
o3 = MWOp(eachchunk(a).chunks[3],steps=stepvectime)

ops = (o1,o2,o3)




rp = ProductArray((o1.steps,o2.steps,o3.steps))

# rangeproduct[3]


inars = (InputArray(a,LoopWindows(rp,Val((1,2,3)))),)


outsize = length.((stepveclat,stepvectime))
o1 = MWOp(DiskArrays.RegularChunks(1,0,outsize[1]))
o2 = MWOp(DiskArrays.RegularChunks(1,0,outsize[2]))
outrp = ProductArray((o1.steps,o2.steps))
outwindows = (LoopWindows(outrp,Val((2,3))),)

function myfunc(x)
    all(ismissing,x) ? (0,zero(eltype(x))) : (1,mean(skipmissing(x)))
end

mainfunc = NonMutatingFunction(myfunc)
function reducefunc((n1,s1),(n2,s2))
  (n1+n2,s1+s2)
end
init = (0,zero(Float64))
filters = (NoFilter(),)
fin(x) = last(x)/first(x)
args = ()
kwargs = (;)
f = UserOp(mainfunc,reducefunc,init,filters,fin,args,kwargs)

optotal = GMDWop(inars, outwindows, f)

struct GMWOPResult{T,N,G<:GMDWop,CS,ISPEC} <: AbstractDiskArray{T,N}
  op::G
  ires::Val{ISPEC}
  chunksize::CS
  max_cache::Float64
  s::NTuple{N,Int}
end
getoutspec(r::GMWOPResult{<:Any,<:Any,<:Any,<:Any,ISPEC}) where ISPEC = r.op.outspecs[ISPEC]
getioutspec(::GMWOPResult{<:Any,<:Any,<:Any,<:Any,ISPEC}) where ISPEC = ISPEC

Base.size(r::GMWOPResult) = length.(getoutspec(r).windows.members)

function DiskArrays.readblock!(res::GMWOPResult, aout,r::AbstractUnitRange...)
  #Find out directly connected loop ranges
  s = res.op.windowsize
  s = Base.OneTo.(s)
  outars = ntuple(_->nothing,length(res.op.outspecs))
  outars = Base.setindex(outars,aout,getioutspec(GMWOPResult))
  outspec = getoutspec(GMWOPResult)
  foreach(getloopinds(outspec),r) do li,ri
    s = Base.setindex(s,ri,li)
  end
  l = length.(s)
  @show s,l
  loopranges = DiskArrays.GridChunks(l,l,offset = first.(s))
  run_loop(res.op,loopranges,outars)
  nothing
end

ow = optotal.outspecs[1]
ow.windows.members

function results_as_diskarrays(o::GMDWop;cs=nothing,max_cache=1e9)
  map(enumerate(o.outspecs)) do i,outspec
    
  end
end



loopranges = DiskArrays.GridChunks(eachchunk(a).chunks[1:2]...,DiskArrays.RegularChunks(120,0,480))
b = zeros(size(a,2),length(stepvectime));
outars = (InputArray(b,outwindows),)
function run_loop(op, loopranges,outars)

  inbuffers_pure = generate_inbuffers(op.inars, loopranges)

  outbuffers = generate_outbuffers(outars,f, loopranges)

  for inow in loopranges
    @show inow
    inbuffers_wrapped = read_range.((inow,),inars,inbuffers_pure);
    outbuffers_now = wrap_outbuffer.((inow,),outars,(f,),outbuffers)
    DiskArrayEngine.run_block(inow, f, inbuffers_wrapped, outbuffers_now)
  
    put_buffer.((inow,), (f,), outbuffers_now, outbuffers, outars)
  end
end





b

using Plots
heatmap(b)





