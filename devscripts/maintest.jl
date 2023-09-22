using DiskArrayEngine
import DiskArrayEngine as DAE
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays, OffsetArrays
#using DiskArrayEngine: MWOp, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
#  create_buffers, read_range, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
#  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, results_as_diskarrays, create_userfunction, steps_per_chunk, apparent_chunksize,
#  find_adjust_candidates, generate_LoopRange, get_loopsplitter, split_loopranges_threads, merge_loopranges_threads, LocalRunner, 
#  merge_outbuffer_collection, DistributedRunner
using StatsBase: rle,mode
using CFTime: timedecode
using Dates
using OnlineStats
using Logging
using Distributed
#global_logger(SimpleLogger(stdout,Logging.Debug))
#global_logger(SimpleLogger(stdout))
using LoggingExtras
using Dagger

using Test



a = zopen("/home/fgans/data/esdc-8d-0.25deg-256x128x128-3.0.2.zarr/air_temperature_2m/", fill_as_missing=true);
t = zopen("/home/fgans/data/esdc-8d-0.25deg-256x128x128-3.0.2.zarr/time", fill_as_missing=true);

a = zopen("/home/fgans/data/esdc-8d-0.25deg-1x720x1440-3.0.2.zarr/air_temperature_2m/", fill_as_missing=true);
t = zopen("/home/fgans/data/esdc-8d-0.25deg-1x720x1440-3.0.2.zarr/time", fill_as_missing=true);

tvec = timedecode(t[:],t.attrs["units"]);
groups = yearmonth.(tvec)

agg1 = DAE.DirectAggregator(DAE.create_userfunction(mean,Union{Float64,Missing}))
agg2 = DAE.ReduceAggregator(DAE.disk_onlinestat(mean))
dimspec = (3=>nothing,)
op1 = DAE.gmwop_for_aggregator(agg1,dimspec,a)
p1 = DAE.optimize_loopranges(op1,5e8)
op2 = DAE.gmwop_for_aggregator(agg2,dimspec,a)
p2 = DAE.optimize_loopranges(op2,5e8)

plan = p1
io = stdout
function printinfo(io::IO,plan::DAE.ExecutionPlan;extended=true)
  n_chunks = length(plan.lr)
  sh_chunks = size(plan.lr)
  mean_windowsize = map(plan.lr.members) do w
    mode(length.(w))
  end
  println(io,"DiskArrayEngine ExecutionPlan")
  println(io,"Processing in $n_chunks blocks of shape $sh_chunks")
  println(io,"With block sizes of approximately $mean_windowsize")
  extended || return nothing
  apc = DAE.access_per_chunk(plan)
  irt = DAE.time_per_chunk(plan)
  tpa = DAE.time_per_array(plan)
  arf = DAE.array_repeat_factor(plan)
  aapc = DAE.actual_access_per_chunk(plan)
  for (ii,ia) in enumerate(plan.input_chunkspecs)
    println(io)
    println(io,"Input Array $ii of size $(last.(last.(ia.cs)))")
    println(io,"Optim Access per chunk: $(apc.input_times[ii])")
    println(io,"Optim time per dim: $(irt.input_times[ii])")
    println(io,"With factor : $(arf.input_times[ii]) resulting in $(tpa.input_times[ii])")
    println(io,"Actual access per chunk: $(aapc.input_times[ii])")
    println(io,"Actual estimated readtime: ")

  end
end
printinfo(stdout,plan)

DAE.access_per_chunk(plan)


p1.sizes_raw
p2.sizes_raw
p1.cost_min/p2.cost_min

using Zarr
p1.lr
cs1 = length.(first.(p1.lr.members[1:2]))
aout1 = zcreate(Float64,size(a)[1:2]...,path=tempname(),fill_value=-1.0e32,chunks=cs1,fill_as_missing=true)
r=DAE.LocalRunner(op1,p1,(aout1,))
run(r)


p2.lr
cs2 = length.(first.(p2.lr.members[1:2]))
aout2 = zcreate(Float64,size(a)[1:2]...,path=tempname(),fill_value=-1.0e32,chunks=cs2,fill_as_missing=true)
r=DAE.LocalRunner(op2,p2,(aout2,))
run(r)

using Plots
heatmap(aout1[:,:])

heatmap(aout2[:,:])

aout2 = zcreate(Float64,90,480,path=tempname(),fill_value=-1.0e32,chunks=cs,fill_as_missing=true)
r=DAE.LocalRunner(op,p,(aout2,))
run(r)

heatmap(aout2)

years, nts = rle(yearmonth.(tvec));
nts;

#cums = [0;cumsum(nts)]

    


#stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(nts)]
#length.(stepvectime)


stepveclat = 1:size(a,2);
stepveclon = 1:size(a,1);
outsteps = outrepfromrle(nts);

outsteps
# rangeproduct[3]

inars = (InputArray(a),);

outars = (create_outwindows((720,480), dimsmap=(2,3),windows = (stepveclat,outsteps)),);

outpath = tempname()

f = disk_onlinestat(Mean)



optotal = GMDWop(inars, outars, f);

# r,  = results_as_diskarrays(optotal);

# r[2:3,2]

lr = DiskArrayEngine.optimize_loopranges(optotal,5e8,tol_low=0.2,tol_high=0.05,max_order=2);






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


