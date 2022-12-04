using Revise
using DiskArrayEngine
using DiskArrays: ChunkType
using Statistics

using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: ProcessingSteps, MWOp, subset_step_to_chunks, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  NonMutatingFunction, create_buffers, read_range, wrap_outbuffer

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


#First example: Latitudinal mean of monthly average
o1 = MWOp(eachchunk(a).chunks[1])
o2 = MWOp(eachchunk(a).chunks[2])
o3 = MWOp(eachchunk(a).chunks[3],steps=stepvectime)

ops = (o1,o2,o3)




rp = ProductArray((o1.steps,o2.steps,o3.steps))

# rangeproduct[3]


inars = (InputArray(a,rp,Val((1,2,3))),)

b = zeros(size(a,2),480);
o1 = MWOp(eachchunk(b).chunks[1])
o2 = MWOp(eachchunk(b).chunks[2])
outrp =  ProductArray((o1.steps,o2.steps))
outars = (InputArray(b,outrp,Val((2,3))),)

function myfunc(x) 
    r = mean(skipmissing(x))
    ismissing(r) ? (0,zero(r)) : (1,r)
end

mainfunc = NonMutatingFunction(myfunc)
function reducefunc((n1,s1),(n2,s2))
  (n1+n2,n2+s2)
end
init = (0,zero(Float32))
filters = (NoFilter(),)
args = ()
kwargs = (;)
f = UserOp(mainfunc,reducefunc,init,filters,args,kwargs)



inbuffers_pure, outbuffers_pure = create_buffers(inars, outars, f, loopranges)


inow = loopranges[9,4,4]
inbuffers_wrapped = read_range.(Ref(inow),inars,inbuffers_pure);
outbuffers_wrapped = wrap_outbuffer.(Ref(inow),outars,outbuffers_pure)

DiskArrayEngine.run_block(inow, f, inbuffers_wrapped, outbuffers_wrapped)

outbuffers_wrapped[1].a

inow


struct WorkerBuffer
  