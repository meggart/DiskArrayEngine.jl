using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: MWOp, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, results_as_diskarrays, create_userfunction, steps_per_chunk, apparent_chunksize,
  find_adjust_candidates, generate_LoopRange, get_loopsplitter, split_loopranges_threads, merge_loopranges_threads, LocalRunner
using StatsBase: rle
using CFTime: timedecode
using Dates
using OnlineStats
using Logging
using Distributed
global_logger(SimpleLogger(stdout))


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


optotal = GMDWop(inars, outwindows, f)


function fit_online!(xout,x,f=identity)
  fit!(xout[],f(x))
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
    args = (preproc,)
)


optotal = GMDWop(inars, outwindows, f)


r, = results_as_diskarrays(optotal)
rsub = r[300:310,200:210]


outwindows = ((lw=LoopWindows(outrp,Val((2,3))),chunks=(nothing, nothing),ismem=true),)
optotal = GMDWop(inars, outwindows, f)
outpath = tempname()
b = zzeros(Float32,size(a,2),length(stepvectime),chunks = (90,480),fill_as_missing=true,path=outpath);



function run_op(op,outars;max_cache=1e8,threaded=true)
  lr = DiskArrayEngine.optimize_loopranges(op,max_cache,tol_low=0.2,tol_high=0.05,max_order=2)
  r = DiskArrayEngine.LocalRunner(optotal,lr,outars,threaded=threaded)
  run(r)
end

@time run_op(optotal, (b,),threaded=true,max_cache=1e9)

using Plots
heatmap(b[:,:])


struct DistributedRunner{OP,LR,OA,IB,OB}
  op::OP
  loopranges::LR
  outars::OA
  threaded::Bool
  inbuffers_pure::IB
  outbuffers::OB
  workers::Vector{Int}
end
function DistributedRunner(op,loopranges,outars;threaded=true,w = workers())
  oplrref = @spawn (op, loopranges)
  makeinbuf = ()->begin
    op, loopranges = fetch(oplrref)
    generate_inbuffers(op.inars, loopranges)
  end
  makeoutbuf = ()->begin
    op, loopranges = fetch(oplrref)
    generate_outbuffers(op.outspecs,op.f, loopranges)
  end
  allinbuffers = Dict(i=>(@spawnat i makeinbuf()) for i in w)
  alloutbuffers = Dict(i=>(@spawnat i makeoutbuf()) for i in w)
  DistributedRunner(op,loopranges,outars, threaded, allinbuffers,alloutbuffers,w)
end

addprocs(2)
lr = DiskArrayEngine.optimize_loopranges(optotal,1e8,tol_low=0.2,tol_high=0.05,max_order=2)
runner = DistributedRunner(optotal, lr, (b,))

function run_loop(runner::DistributedRunner,loopranges = runner.loopranges;groupspecs=nothing)
  if groupspecs !== nothing && :output_chunk in groupspecs
    piddir = @spawn tempname()
  else
    piddir = nothing
  end
  getbuffers = ()->runner,fetch(runner.inbuffers_pure[myid()]),fetch(runner.outbuffers[myid()])
  pmap_with_data(loopranges,initfunc=getbuffers) do inow,prep
    @debug "inow = ", inow
    runner,inbuffers_pure,outbuffers = prep
    inbuffers_wrapped = read_range.((inow,),runner.op.inars,inbuffers_pure);
    outbuffers_now = wrap_outbuffer.((inow,),runner.outars,runner.op.outspecs,runner.op.f.init,runner.op.f.buftype,outbuffers)
    run_block(runner.op,inow,inbuffers_wrapped,outbuffers_now,runner.threaded)
    put_buffer.((inow,),runner.op.f.finalize, outbuffers_now, runner.outbuffers, runner.outars, (piddir,))
  end
  if groupspecs !== nothing && :reducedim in groupspecs
    @debug "Merging buffers"
  end
end

inow = (91:180,631:720,1:480)

lr = DiskArrayEngine.optimize_loopranges(optotal,3e7,tol_low=0.2,tol_high=0.05,max_order=2)

outars= (b,)


using DiskArrayEngine: get_procgroups






using Distributed
addprocs(2)
workerpool = WorkerPool([2])
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



using Interpolations: Interpolations, weightedindexes, itpinfo, value_weights, InterpGetindex, coefficients, tweight, tcoef, prefilter, 
  copy_with_padding, prefilter!, degree, prefiltering_system, popwrapper

A_x1 = 1:.1:10
A_x2 = 1:.5:20
f(x1, x2) = log(x1+x2)
A = [f(x1,x2) for x1 in A_x1, x2 in A_x2]
A[1:2, :] .= NaN
A[:, 1:2] .= NaN


it = BSpline(Cubic(Line(OnGrid())))
it = BSpline(Linear())
ret = copy_with_padding(Float64, A, it)
@which prefilter!(Float64, ret, it)

@which prefilter(tweight(A), tcoef(A), A, it)

sz = size(ret)
first = true
#for dim in 1:ndims(ret)
dim = 1
M, b = prefiltering_system(Float64, Float64, sz[dim], degree(it))
popwrapper(ret)

@which Interpolations.A_ldiv_B_md!(popwrapper(ret), M, popwrapper(ret), dim, b)
#end
    ret

Apad = prefilter(tweight(A), tcoef(A), A, it)



@which interpolate(, ,A, it)




itp = interpolate(A, BSpline(Linear()))

x  =(1.5,2.5)

wis = weightedindexes((value_weights,), itpinfo(itp)..., x)

@which coefficients(itp)

InterpGetindex(itp)

@which InterpGetindex(itp)


itpgi[wis...]

interp_getindex(A.coeffs, ntuple(_ -> 0, Val(N)), map(indexflag, I)...)