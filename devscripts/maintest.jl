using Revise
using DiskArrayEngine
using DiskArrays: ChunkType
using Statistics

using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: ProcessingSteps, MWOp, subset_step_to_chunks, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  NonMutatingFunction, create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer

a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

t = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/time/", fill_as_missing=true)[:]

stepvectime = [(i*46-45):i*46 for i in 1:40]


#tvec = (Date(1980,1,1).+Day.(Int.(t.-1)))
#acclens = map(t->(Date(year(t),month(t),1)),tvec) |> unique .|> daysinmonth |> cumsum
#starts = [0;acclens[1:end-1]]
#stepvectime = ProcessingSteps(0,range.(starts.+1,acclens))
stepveclat = ProcessingSteps(0,1:size(a,2))
stepveclon = ProcessingSteps(0,1:size(a,1))


loopsizeall = (1440,720,40)
#Chunks over the loop indices
loopranges = DiskArrays.GridChunks(eachchunk(a).chunks[1:2]...,DiskArrays.RegularChunks(10,0,40))


#First example: Latitudinal mean of annual average
o1 = MWOp(eachchunk(a).chunks[1])
o2 = MWOp(eachchunk(a).chunks[2])
o3 = MWOp(eachchunk(a).chunks[3],steps=stepvectime)

ops = (o1,o2,o3)




rp = ProductArray((o1.steps,o2.steps,o3.steps))

# rangeproduct[3]


inars = (InputArray(a,rp,Val((1,2,3))),)

b = zeros(size(a,2),40);
o1 = MWOp(eachchunk(b).chunks[1])
o2 = MWOp(eachchunk(b).chunks[2])
outrp =  ProductArray((o1.steps,o2.steps))
outars = (InputArray(b,outrp,Val((2,3))),)

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

inbuffers_pure = generate_inbuffers(inars, loopranges)

outbuffers = generate_outbuffers(outars,f, loopranges)


for inow in loopranges
  @show inow
  inbuffers_wrapped = read_range.((inow,),inars,inbuffers_pure);
  outbuffers_now = wrap_outbuffer.((inow,),outars,(f,),outbuffers)

  DiskArrayEngine.run_block(inow, f, inbuffers_wrapped, outbuffers_now)

  put_buffer.((inow,), (f,), outbuffers_now, outbuffers, outars)
end

b

using Plots
heatmap(b)

outbuffers_now[1].a


obw = outbuffers_wrapped[1]

a[:,1,:]
b[:,1]