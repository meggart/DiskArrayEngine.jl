using DiskArrayEngine
import DiskArrayEngine as DAE
using DiskArrays: ChunkType, RegularChunks
using Statistics
using Zarr, DiskArrays
using StatsBase: rle, mode
using CFTime: timedecode
using Dates
using OnlineStats
using Logging
using Distributed
using LoggingExtras
using Test
using DataStructures: OrderedSet




inwindows1 = DAE.MovingWindow(1, 5, 5, 4)
outwindows1 = 1:4
inwindows2 = DAE.MovingWindow(1, 2, 2, 2)
outwindows2 = 1:2

inar = DAE.InputArray(1:20, windows=(inwindows1,))
outspecs = (DAE.create_outwindows(4, windows=(outwindows1,)),)
f = create_userfunction(sum, Float64)
op1 = DAE.GMDWop((inar,), outspecs, f)
r = results_as_diskarrays(op1)[1]

inar2 = DAE.InputArray(r, windows=(inwindows2,))
outspecs2 = (DAE.create_outwindows(4, windows=(outwindows2,)),)
f2 = create_userfunction(sum, Float64)
op2 = DAE.GMDWop((inar2,), outspecs2, f2)
r2 = results_as_diskarrays(op2)[1]

g = DAE.result_to_graph(r2)

@test length(g.nodes) == 3
@test g.nodes[1] == DAE.MwopOutNode(false, nothing, (2,), Float64)
@test g.nodes[2] == DAE.MwopOutNode(false, nothing, (4,), Float64)
@test g.nodes[3] == 1:20

@test length(g.connections) == 2
conn1, conn2 = g.connections
@test conn1.inputids == [3]
@test conn1.outputids == [2]
@test conn2.inputids == [2]
@test conn2.outputids == [1]

nodemergestrategies = DAE.collect_strategies(g)

@test only(nodemergestrategies[2]) isa DAE.BlockMerge
@test nodemergestrategies[1] == [nothing]
@test nodemergestrategies[3] == [nothing]

dimmap = DAE.create_loopdimmap(conn1, conn2, 2)
@test dimmap isa DAE.DimMap
@test dimmap.d == Dict(1 => 1)

newop = DAE.merge_operations(DAE.BlockMerge, conn1, conn2, 2, dimmap)
@test newop isa DAE.UserOp
@test newop.f isa DAE.BlockFunctionChain
@test newop.f.funcs[1] === f.f
@test newop.f.funcs[2] === f2.f
@test newop.f.args == [((1,), (1,)), ((2,), (2,))]
@test newop.f.transfers == [1 => [2]]

newconn, newnodes = DAE.merged_connection(DAE.BlockMerge, g, conn1, conn2, 2, newop, nodemergestrategies, dimmap)

@test newconn isa DAE.MwopConnection
@test newconn.f === newop
@test newconn.inputids == [3, 4]
@test newconn.outputids == [2, 1]

win1 = newconn.inwindows[1].windows.members[1]
@test win1 isa DAE.Window
@test eltype(win1) <: DAE.WindowGroup
@test length(win1) == 2
@test win1[1].g == 1:2
@test win1[1].parent == [1:5, 6:10, 11:15, 16:20]
@test win1[2].g == 3:4
@test win1[2].parent == [1:5, 6:10, 11:15, 16:20]
@test DAE.avg_step(win1) == 10
@test DAE.max_size(win1) == 10


wout1 = newconn.outwindows[1].windows.members[1]
@test wout1 isa DAE.Window
@test eltype(wout1) <: DAE.WindowGroup
@test length(wout1) == 2
@test wout1[1].g == 1:2
@test wout1[1].parent == 1:4
@test wout1[2].g == 3:4
@test wout1[2].parent == 1:4
@test DAE.avg_step(wout1) == 2
@test DAE.max_size(wout1) == 2

win2 = newconn.inwindows[2].windows.members[1]
@test win2 isa DAE.Window
@test eltype(win2) <: DAE.WindowGroup
@test length(win2) == 2
@test win2[1].g == 1:1
@test win2[1].parent == [1:2, 3:4]
@test win2[2].g == 2:2
@test win2[2].parent == [1:2, 3:4]
@test DAE.avg_step(win2) == 2
@test DAE.max_size(win2) == 2

wout2 = newconn.outwindows[2].windows.members[1]
@test wout2 isa DAE.Window
@test eltype(wout2) <: DAE.WindowGroup
@test length(wout2) == 2
@test wout2[1].g == 1:1
@test wout2[1].parent == 1:2
@test wout2[2].g == 2:2
@test wout2[2].parent == 1:2
@test DAE.avg_step(wout2) == 1
@test DAE.max_size(wout2) == 1

@test length(newnodes) == 1
@test newnodes[1] == DAE.EmptyInput{Float64,1}((4,))

append!(g.nodes, newnodes)

deleteat!(g.connections, [1, 2])
push!(g.connections, newconn)

newop = DAE.gmwop_from_reducedgraph(g)

inar = newop.inars[1]
cspec = DAE.get_chunkspec(inar, (2,))
@test cspec.app_cs == (2,)
@test cspec.windowfac == (10,)

lr = DAE.custom_loopranges(newop, (1,))

runner = DAE.LocalRunner(newop, lr)



g = zopen("https://s3.bgc-jena.mpg.de:9000/esdl-esdc-v2.1.1/esdc-8d-0.25deg-184x90x90-2.1.1.zarr")


g = zopen(joinpath(homedir(), "data/esdc-8d-0.25deg-256x128x128-3.0.2.zarr/"), fill_as_missing=true);
a = g["air_temperature_2m"];
t = g["time"]

tvec = timedecode(t[:], t.attrs["units"]);
groups = yearmonth.(tvec)

r = aggregate_diskarray(a, mean, (1 => nothing, 2 => 8, 3 => groups), strategy=:direct)

#a = compute(r)

r2 = aggregate_diskarray(r, maximum, (2 => nothing,))



r3 = r .+ 273.15

finalres = r2 .+ r3

finalres[1, 45, 100]

g = DAE.MwopGraph()
outnode = DAE.to_graph!(g, r2);

DAE.remove_aliases!(g)

using CairoMakie, GraphMakie
p = graphplot(g, elabels=DAE.edgenames(g), ilabels=DAE.nodenames(g))

#DAE.fuse_step_direct!(g)




function gmwop_from_conn(conn)
  op = conn.f
  inputs = InputArray.(g.nodes[conn.inputids], conn.inwindows)
  outspecs = map(g.nodes[conn.outputids], conn.outwindows) do outnode, outwindow
    (; lw=outwindow, chunks=outnode.chunks, ismem=outnode.ismem)
  end
  DAE.GMDWop(inputs, outspecs, op)
end


using Graphs: nv, outneighbors, inneighbors
ioutnodes = (findall(n -> !isempty(inneighbors(g, n)) && isempty(outneighbors(g, n)), 1:nv(g)))
lastop = findall(conn -> all(in(conn.outputids), ioutnodes), g.connections) |> only
op = gmwop_from_conn(g.connections[lastop])
rgraph = results_as_diskarrays(op)[1]

runner = rgraph[1, 45, 100]




gold = deepcopy(g)


gold.connections[1].inputids
gold.connections[1].outputids
gold.connections[1].inwindows[1]
gold.connections[1].outwindows[1]

gold.connections[2].inwindows[1]
gold.connections[2].inwindows[3]
gold.connections[2].outwindows[1]


DAE.fuse_step_block!(g)

remaining_conn = only(g.connections)


remaining_conn.inputids
remaining_conn.outputids

remaining_conn.inwindows[1]
remaining_conn.inwindows[2]
remaining_conn.inwindows[3]
remaining_conn.inwindows[4]

remaining_conn.outwindows[1]
remaining_conn.outwindows[2]



g.nodes[7]

p = graphplot(g, ilabels=DAE.nodenames(g))

remaining_conn = only(g.connections)

remaining_conn.inputids
remaining_conn.outputids
remaining_conn.outwindows[1]




rnow = DAE.results_as_diskarrays(mergedop)[2]


rnow[45, 100]




using Logging
mylogger = EarlyFilteredLogger(SimpleLogger(Logging.Debug)) do log
  (log._module == DiskArrayEngine && log.level >= Logging.Debug) || log.level >= Logging.Info
end
global_logger(mylogger)

lr = DAE.optimize_loopranges(mergedop, 5e8, tol_low=0.2, tol_high=0.05, max_order=2);
lr

outar = zeros(Union{Missing,Float32}, 90, 516)

runner = DAE.LocalRunner(mergedop, lr, (nothing, outar));

run(runner)

outar[45, 100]

using Makie, CairoMakie
heatmap(outar)







struct EnvFloat{T} <: AbstractFloat
  val::T
  mask::UInt8
end
Base.convert(T::Type{<:Number}, v::EnvFloat) = convert(T, v.val)
Base.convert(::Type{EnvFloat}, x::Number) = EnvFloat(AbstractFloat(x), 0x00)
Base.convert(::Type{EnvFloat{T}}, x::Number) where {T} = EnvFloat(convert(T, x), 0x00)
Base.convert(::Type{EnvFloat{T}}, x::EnvFloat) where {T} = EnvFloat(convert(T, x.val), x.mask)
Base.promote_rule(::Type{EnvFloat{T}}, ::Type{S}) where {T,S<:Number} = EnvFloat{promote_type(T, S)}

ef = EnvFloat(3.0, 0x04)



1



DAE.eliminate_node(dg.nodegraph, i_eliminate, DAE.DirectMerge())

remaining_conn = only(dg.nodegraph.connections)

op = remaining_conn.f
inputs = InputArray.(dg.nodegraph.nodes[remaining_conn.inputids], remaining_conn.inwindows)
outspecs = map(dg.nodegraph.nodes[remaining_conn.outputids], remaining_conn.outwindows) do outnode, outwindow
  (; lw=outwindow, chunks=outnode.chunks, ismem=outnode.ismem)
end
mergedop = DAE.GMDWop(inputs, outspecs, op)

lr = DAE.optimize_loopranges(mergedop, 5e8, tol_low=0.2, tol_high=0.05, max_order=2)
outar = zeros(Float32, 90, 516)
runner = DAE.LocalRunner(mergedop, lr, (outar,))
# inow = (1:1, 1:16, 1:67)
# inbuffers_wrapped = DAE.read_range.((inow,),mergedop.inars,runner.inbuffers_pure);
# outbuffers_now = DAE.extract_outbuffer.((inow,),mergedop.outspecs,mergedop.f.init,mergedop.f.buftype,runner.outbuffers)
# DAE.run_block(mergedop,inow,inbuffers_wrapped,outbuffers_now,true)
# DAE.put_buffer.((inow,),mergedop.f.finalize, outbuffers_now, runner.outbuffers, runner.outars,nothing)
# runner.inbuffers_pure

run(runner)

outar

heatmap(outar)

p = graphplot(g, elabels=DAE.edgenames(g), ilabels=DAE.nodenames(g))




#First decide if merge of functions can be done ny element or has to be done
# by block. Then define possible block boundaries and if not possible resort to
# the whole dimension



aout = DAE.compute(r)




using CairoMakie
heatmap(aout)


using Plots
heatmap(aout1[:, :])

heatmap(aout2[:, :])

aout2 = zcreate(Float64, 90, 480, path=tempname(), fill_value=-1.0e32, chunks=cs, fill_as_missing=true)
r = DAE.LocalRunner(op, p, (aout2,))
run(r)

heatmap(aout2)

years, nts = rle(yearmonth.(tvec));
nts;

#cums = [0;cumsum(nts)]




#stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(nts)]
#length.(stepvectime)


stepveclat = 1:size(a, 2);
stepveclon = 1:size(a, 1);
outsteps = outrepfromrle(nts);

outsteps
# rangeproduct[3]

inars = (InputArray(a),);

outars = (create_outwindows((720, 480), dimsmap=(2, 3), windows=(stepveclat, outsteps)),);

outpath = tempname()

f = disk_onlinestat(Mean)



optotal = GMDWop(inars, outars, f);

# r,  = results_as_diskarrays(optotal);

# r[2:3,2]

lr = DiskArrayEngine.optimize_loopranges(optotal, 5e8, tol_low=0.2, tol_high=0.05, max_order=2);






#Test for time to extract series of longitudes
cs = 100
function extract_slice(a, cs)
  r = zeros(Union{Missing,Float32}, 1440)
  for i in 1:cs:1440
    r[i:min(1440, i + cs - 1)] .= a[i:min(1440, i + cs - 1)]
  end
  r
end
csvec = [10:90; 95:5:200]
using Plots
readtime = [@elapsed extract_slice(a, cs) for cs in csvec]
p = plot(csvec, readtime, log="x")
ticvec = [15, 18, 20, 30, 36, 45, 60, 90, 120, 135, 150, 180]
xticks!(p, ticvec)
vline!(p, ticvec)

using DiskArrays: approx_chunksize
using DiskArrayEngine: RegularWindows

singleread = median([@elapsed a[first(eachchunk(a))...] for _ in 1:10])

#p = plot(csvec,integrated_readtime.((eachchunk(a).chunks[1],),singleread,csvec))
#plot!(p,csvec,readtime)


#2 example arrays
p1 = tempname()
p2 = tempname()
a1 = zcreate(Float32, 10000, 10000, path=p1, chunks=(10000, 1), fill_value=2.0, fill_as_missing=false)
a2 = zcreate(Float32, 10000, 10000, path=p2, chunks=(1, 10000), fill_value=5.0, fill_as_missing=false)



eltype(r)

size(r)

rp = ProductArray((1:10000, DiskArrayEngine.RegularWindows(1, 10000, step=3)))

# rangeproduct[3]
inars = (InputArray(a1, LoopWindows(rp, Val((1, 2)))), InputArray(a2, LoopWindows(rp, Val((1, 2)))))

outrp = ProductArray(())
outwindows = ((lw=LoopWindows(outrp, Val(())), chunks=(), ismem=false),)

f = create_userfunction(
  +,
  Float64,
  red=+,
  init=0.0,
)

optotal = GMDWop(inars, outwindows, f)

DiskArrayEngine.optimize_loopranges(optotal, 1e8)

compute_time(window, arraychunkspec)
compute_bufsize(window, arraychunkspec)
all_constraints(window, arraychunkspec)

using Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
using ForwardDiff, ModelingToolkit
window = [1000, 1000]
loopsize = (10000, 10000)



import DiskArrayEngine as DAE
using Zarr, Test


using FFTW, DataFrames, GLM
function ft_by_reg(x)
  len = length(x)
  time = range(1.0, len)
  max_norm_freq = len ÷ 2
  return ft_by_reg(time, x, max_norm_freq)
end

function ft_by_reg(time, x, max_norm_freq)
  timespan = maximum(time) - minimum(time)
  normtime = (time .- minimum(time)) ./ timespan
  function slope(y, time, freq)
    xsin = sin.(2π * time * freq)
    xcos = cos.(2π * time * freq)
    df = DataFrame(; xsin, xcos, y)
    lm(@formula(y ~ 1 + xsin + xcos), df) |> coef
  end
  freqs = 1:max_norm_freq
  coefs = [slope(x, normtime, fr) for fr in freqs]

  return coefs

end

# Test it

# Make wave with distinct frequencies...
len = 1000

sincoeffs = rand(100)
coscoeffs = rand(100)
freqs = 1 ./ rand(1:len, len ÷ 10)

time = range(1.0, len)
trifun(x, freq, coeffs) = coeffs[1] * sin(2π * x * freq) + coeffs[2] * cos(2π * x * freq)
constr(x, freqs, sincoeffs, coscoeffs) = mapreduce((f, c) -> trifun(x, f, c), +, freqs, zip(sincoeffs, coscoeffs))
x = [constr(t, freqs, sincoeffs, coscoeffs) for t in time]
x = cumsum(randn(len))  #... or just time a random walk 

fft_spec = rfft(x) .|> abs
my_spec = map(ft_by_reg(x)) do r
  sqrt(r[2:3] .|> abs2 |> sum)
end # sqrt(a²+b²), where a, b are the cos, sin coefs

xmiss = x |> allowmissing
xmiss[rand(1:len, len ÷ 10)] .= missing # 10% missing
my_spec_miss = map(ft_by_reg(xmiss)) do r
  sqrt(r[2:3] .|> abs2 |> sum)
end # sqrt(a²+b²), where a, b are the cos, sin coefs

ok = findall(x -> x > 0.01, my_spec)
scatter(fft_spec[2:end][ok], my_spec[ok])
lines(fft_spec[2:end][ok] ./ my_spec[ok])
lines(my_spec_miss ./ my_spec)
lines(my_spec_miss ./ fft_spec[2:end])