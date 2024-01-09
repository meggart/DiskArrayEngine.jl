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
1
using Test
using DataStructures: OrderedSet

g = zopen("/home/fgans/data/esdc-8d-0.25deg-256x128x128-3.0.2.zarr/",fill_as_missing=true);
a = g["air_temperature_2m"];
t = g["time"]

tvec = timedecode(t[:],t.attrs["units"]);
groups = yearmonth.(tvec)

r = aggregate_diskarray(a,mean,(1=>nothing,2=>8,3=>groups))

r2 = aggregate_diskarray(r,maximum,(2=>nothing,))

r3 = r .- 5

finalres = r2 .+ r3


g = DAE.MwopGraph()
DAE.to_graph!(g,finalres.op);
map(i->(i.inputids,i.outputids),g.connections)
DAE.remove_aliases!(g)
using CairoMakie, GraphMakie
p = graphplot(g,elabels=DAE.edgenames(g),ilabels=DAE.nodenames(g))
map(i->(i.inputids,i.outputids),g.connections)

inops = []
outops = []
node_to_merge = 3
for c in g.connections
  iin = findfirst(==(node_to_merge),c.outputids)
  iout = findfirst(==(node_to_merge),c.inputids)
  !isnothing(iin) && push!(inops,(iin,c))
  !isnothing(iout) && push!(outops,(iout,c))
end

#node_dimension_map maps the dimensions from every input op

i,conn = first(inops)
nodedim = DAE.getsubndims(conn.outwindows[i])
innodemaps = map(inops) do (i,conn)
  inds = DAE.getloopinds(conn.outwindows[i])
  Dict(j=>i for (i,j) in enumerate(inds))
end
outnodemaps = map(outops) do (i,conn)
  inds = DAE.getloopinds(conn.inwindows[i])
  Dict(j=>i for (i,j) in enumerate(inds))
end
dimtomerge = 1
inwindows = map(inops,innodemaps) do map
  ii = findfirst(==(dimtomerge),innodemaps)
  if ii !== nothing
    

conn.outputids[i]

  

aout = DAE.compute(r)




using CairoMakie
heatmap(aout)


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
using Plots
readtime = [@elapsed extract_slice(a,cs) for cs in csvec]
p = plot(csvec,readtime,log="x")
ticvec = [15,18,20,30,36,45,60,90,120,135,150,180]
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



import DiskArrayEngine as DAE
using Zarr, Test


using FFTW, DataFrames, GLM
function ft_by_reg(x)
    len=length(x)
    time=range(1.0, len)
    max_norm_freq = len ÷ 2
    return ft_by_reg(time, x, max_norm_freq)
 end

 function ft_by_reg(time, x, max_norm_freq)
    timespan = maximum(time) - minimum(time)
    normtime = (time .- minimum(time)) ./ timespan
    function slope(y, time, freq)
        xsin = sin.(2π*time*freq)
        xcos = cos.(2π*time*freq)
        df=DataFrame(;xsin,xcos, y)
        lm(@formula(y~1+xsin+xcos), df) |> coef
    end
    freqs = 1:max_norm_freq
    coefs=[slope(x, normtime, fr) for fr in freqs]

    return coefs

 end

 # Test it

# Make wave with distinct frequencies...
len=1000

sincoeffs = rand(100)
coscoeffs = rand(100)
freqs=1 ./ rand(1:len, len ÷ 10)

time = range(1.0, len)
trifun(x, freq, coeffs) = coeffs[1] * sin(2π * x * freq) + coeffs[2] * cos(2π * x * freq)
constr(x, freqs, sincoeffs, coscoeffs) = mapreduce((f,c)->trifun(x,f,c), +, freqs, zip(sincoeffs, coscoeffs))
x = [constr(t, freqs, sincoeffs, coscoeffs) for t in time]
x = cumsum(randn(len))  #... or just time a random walk 

fft_spec=rfft(x) .|> abs
 my_spec = map(ft_by_reg(x)) do r sqrt(r[2:3] .|> abs2 |> sum) end # sqrt(a²+b²), where a, b are the cos, sin coefs

 xmiss = x |> allowmissing
 xmiss[rand(1:len, len ÷ 10)] .= missing # 10% missing
 my_spec_miss = map(ft_by_reg(xmiss)) do r sqrt(r[2:3] .|> abs2 |> sum) end # sqrt(a²+b²), where a, b are the cos, sin coefs

 ok = findall(x->x>0.01, my_spec)
 scatter(fft_spec[2:end][ok],my_spec[ok])
 lines(fft_spec[2:end][ok] ./  my_spec[ok])
 lines(my_spec_miss ./ my_spec)
 lines(my_spec_miss ./ fft_spec[2:end])