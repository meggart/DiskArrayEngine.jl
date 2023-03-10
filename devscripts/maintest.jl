using Revise
using DiskArrayEngine
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays, OffsetArrays
using DiskArrayEngine: MWOp, PickAxisArray, internal_size, ProductArray, InputArray, getloopinds, UserOp, mysub, ArrayBuffer, NoFilter, AllMissing,
  create_buffers, read_range, wrap_outbuffer, generate_inbuffers, generate_outbuffers, get_bufferindices, offset_from_range, generate_outbuffer_collection, put_buffer, 
  Output, _view, Input, applyfilter, apply_function, LoopWindows, GMDWop, results_as_diskarrays, create_userfunction, steps_per_chunk, apparent_chunksize,
  find_adjust_candidates, generate_LoopRange, get_loopsplitter, split_loopranges_threads, merge_loopranges_threads
using StatsBase: rle
using CFTime: timedecode
using Dates
using OnlineStats
using Logging
global_logger(SimpleLogger(stdout, Logging.Debug))


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
b = zzeros(Float32,size(a,2),length(stepvectime),chunks = (100,480),fill_as_missing=true);



function run_op(op,outars;max_cache=1e8,threaded=true)
  lr = DiskArrayEngine.optimize_loopranges(op,1e8,tol_low=0.2,tol_high=0.05,max_order=2)
  DiskArrayEngine.run_loop(optotal,lr,outars,threaded=true)
end

@time run_op(optotal, (b,),threaded=false,max_cache=5e8)

inow = (91:180,631:720,1:480)

lr = DiskArrayEngine.optimize_loopranges(optotal,2e8,tol_low=0.2,tol_high=0.05,max_order=2)

outars= (b,)


function is_output_chunk_overlap(spec,outar,idim)
  li = getloopinds(spec)
  if idim in li
    ii = findfirst(==(idim),li)
    loopind = li[ii]
    cs = eachchunk(outar).chunks[ii]
    chunkbounds = cumsum(length.(cs))
    !all(in(chunkbounds),cumsum(length.(lr.members[loopind])))
  else
    false
  end
end
function is_output_reducedim(spec,outar,idim)
  li = getloopinds(spec)
  !in(idim,li)
end

function split_dim_reasons(op,lr,outars)
  ret = ntuple(_->Symbol[],ndims(lr))
  for (spec,ar) in zip(op.outspecs,outars)
    foreach(1:ndims(lr)) do idim
      if is_output_chunk_overlap(spec,ar,idim)
        push!(ret[idim],:output_chunk)
      end
      if is_output_reducedim(spec,ar,idim)
        push!(ret[idim],:reducedim)
      end
    end
  end
  ret
end

spr = split_dim_reasons(optotal,lr,outars)
allsplits = unique(filter(!isempty,spr))


map(i->(i...,),spr)
optotal.windowsize

reason_priority = Dict(
  :foldl => 1, 
  :output_chunk => 2, 
  :reducedim => 3,
  :overlapinputs =>4,
)

optotal

struct GroupLoopDim
  reasons
  dims
end

abstract type ProcessingGroup end
struct RootGroup{LW}
  lw::LW
end
group_n_jobs(g::RootGroup) = length(g.lw)
group_job_indices(g::RootGroup) = eachindex(g.lw)
group_job_getindex(g::RootGroup,i) = g.lw[i]

function run_process_group(g::RootGroup, workers, op,loopranges,outars;threaded = true)
  inbuffers_pure = generate_inbuffers(op.inars, loopranges)
  
  outbuffers = generate_outbuffers(op.outspecs,op.f, loopranges)
  
  for inow in loopranges
    @debug "inow = ", inow
    inbuffers_wrapped = read_range.((inow,),op.inars,inbuffers_pure);
    outbuffers_now = wrap_outbuffer.((inow,),outars,op.outspecs,op.f.init,op.f.buftype,outbuffers)
    @debug "Axes of wrapped input buffers"
    run_block(op,inow,inbuffers_wrapped,outbuffers_now,threaded)
    put_buffer.((inow,),op.f.finalize, outbuffers_now, outbuffers, outars)
  end
end


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