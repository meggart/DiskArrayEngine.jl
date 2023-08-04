using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: MWOp, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  create_buffers, read_range, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, results_as_diskarrays, create_userfunction, steps_per_chunk, apparent_chunksize,
  find_adjust_candidates, generate_LoopRange, get_loopsplitter, split_loopranges_threads, merge_loopranges_threads, LocalRunner, 
  merge_outbuffer_collection, DistributedRunner
using StatsBase: rle
using CFTime: timedecode
using Dates
using OnlineStats
using Logging
using Distributed
#global_logger(SimpleLogger(stdout,Logging.Debug))
#global_logger(SimpleLogger(stdout))


a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

t = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/time/", fill_as_missing=true)
tvec = timedecode(t[:],t.attrs["units"])
years, nts = rle(yearmonth.(tvec))
nts

#cums = [0;cumsum(nts)]
function outrepfromrle(nts)
  r = Int[]
  for i in 1:length(nts)
    for _ in 1:nts[i]
      push!(r,i)
    end
  end
  r
end
    


#stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(nts)]
#length.(stepvectime)


stepveclat = 1:size(a,2)
stepveclon = 1:size(a,1)
outsteps = outrepfromrle(nts)



# rangeproduct[3]

inars = (InputArray(a),)

outars = (create_outwindows((720,480), dimsmap=(2,3),windows = (stepveclat,outsteps)),)

outpath = tempname()
b = zzeros(Float32,size(a,2),length(outsteps),chunks = (90,480),fill_as_missing=true,path=outpath);

function fit_online!(xout,x,f=identity)
  fx = f(x)
  ismissing(fx) || fit!(xout[],f(x))
end
preproc(x) = mean(skipmissing(x))
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
#    args = (preproc,)
)

optotal = GMDWop(inars, outars, f)

r,  = results_as_diskarrays(optotal)



r[2:3,300]


lr = DiskArrayEngine.optimize_loopranges(optotal,1e8,tol_low=0.2,tol_high=0.05,max_order=2)
lr = ProductArray((RegularChunks(15,0,50),lr.members[2:3]...))

out = zeros(Union{Float32,Missing},720,480)
runner = LocalRunner(optotal,lr,(out,),threaded=true)


@profview run(runner)


# function myfunc(x)
#   all(ismissing,x) ? (0,zero(eltype(x))) : (1,mean(skipmissing(x)))
# end

# function reducefunc((n1,s1),(n2,s2))
#   (n1+n2,s1+s2)
# end
# init = ()->(0,zero(Float64))
# filters = (NoFilter(),)
# fin(x) = last(x)/first(x)
# outtypes = (Union{Float32,Missing},)
# args = ()
# kwargs = (;)
# f = create_userfunction(
#   myfunc,
#   Union{Float32,Missing},
#   red = reducefunc, 
#   init = init, 
#   finalize=fin,
#   buftype = Tuple{Int,Union{Float32,Missing}},  
# )


# optotal = GMDWop(inars, outwindows, f)





# r, = results_as_diskarrays(optotal)
# rsub = r[300:310,200:210]

outpath = tempname()
b = zzeros(Float32,size(a,2),length(stepvectime),chunks = (90,480),fill_as_missing=true,path=outpath);




# function run_op(op,outars;max_cache=1e8,threaded=true)
#   lr = DiskArrayEngine.optimize_loopranges(op,max_cache,tol_low=0.2,tol_high=0.05,max_order=2)
#   r = DiskArrayEngine.LocalRunner(optotal,lr,outars,threaded=threaded)
#   run(r)
# end

# @time run_op(optotal, (b,),threaded=true,max_cache=1e9)

# using Plots
# heatmap(b[:,:])




rmprocs(workers())
addprocs(2)
@everywhere begin
  using DiskArrayEngine, Zarr, OnlineStats
  function fit_online!(xout,x,f=identity)
    fit!(xout[],f(x))
  end
  preproc(x) = mean(skipmissing(x))
  init = ()->OnlineStats.Mean()
  fin_onine(x) = nobs(x) == 0 ? missing : OnlineStats.value(x)
end
f = create_userfunction(
    fit_online!,
    Float64,
    is_mutating = true,
    red = OnlineStats.merge!, 
    init = init, 
    finalize=fin_onine,
    buftype = Mean,  
    args = (preproc,)
)
optotal = GMDWop(inars, outwindows, f)

lr = DiskArrayEngine.optimize_loopranges(optotal,1e8,tol_low=0.2,tol_high=0.05,max_order=2)
runner = DistributedRunner(optotal, lr, (b,))
groups = DiskArrayEngine.get_procgroups(runner.op, runner.loopranges, runner.outars)
sch = DiskArrayEngine.DiskEngineScheduler(groups, runner.loopranges, runner)
DiskArrayEngine.run_group(sch)

r = runner.inbuffers_pure[2] |> fetch;

inow = (91:180,631:720,1:480)

lr = DiskArrayEngine.optimize_loopranges(optotal,3e7,tol_low=0.2,tol_high=0.05,max_order=2)

outars= (b,)


using DiskArrayEngine: get_procgroups

using Distributed
addprocs(2)

@everywhere begin

end


data = ()->[1,2,3]
p = DataPool(workers(),data)

pmap(p,1:10) do data,i
  println(i)
  sum(data)
end

using Distributed
addprocs(2)
workerpool = CachingPool([2])
push!(workerpool,3)
@everywhere function distrtest(i)
  println(i, " ", myid())
  sleep(1)
end
r = @async pmap(distrtest, workerpool, 1:100)
addprocs(2)
@everywhere function distrtest(i)
  println(i, " ", myid())
  sleep(1)
end
push!(workerpool,4)
push!(workerpool,5)





struct ReducedimsGroup{P,N}
  parent::P
  dims::NTuple{N,Int}
  is_foldl::Bool

end

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


