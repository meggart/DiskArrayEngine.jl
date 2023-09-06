using DiskArrayEngine
import DiskArrayEngine as DAE
using Zarr
using DiskArrays: eachchunk
using Distributed
addprocs(4,exeflags=["--project=$(@__DIR__)"])
@everywhere begin
    using DiskArrayEngine, Zarr,LoggingExtras
    # mylogger = EarlyFilteredLogger(FileLogger("$(myid()).log")) do log
    #     (log._module == DiskArrayEngine && log.level >= Logging.Debug) || log.level >=Logging.Info
    # end
    # global_logger(mylogger)  
end


a = zopen("/home/fgans/data/esdc-8d-0.25deg-1x720x1440-3.0.2.zarr/gross_primary_productivity/", fill_as_missing=true);

isdir("rechunked.zarr") && rm("./rechunked.zarr",recursive=true)
aout = zcreate(eltype(a),size(a)...,path="./rechunked.zarr",chunks = (90,90,184),fill_value=-1.32f0,fill_as_missing=true)

max_cache=5e8
ain = a

size(aout) == size(ain) || throw(ArgumentError("Input and Output arrays must have the same size"))
inar = InputArray(ain)

outar = create_outwindows(size(aout),ismem=false,chunks=eachchunk(aout))

f = create_userfunction(DAE.copydata_rechunk!,eltype(aout),is_blockfunction=true,is_mutating=true)

op = GMDWop((inar,), (outar,), f);

lb = [0.0,map(_->1.0,op.windowsize)...]
ub = [max_cache,op.windowsize...]
x0 = [2.0 for _ in op.windowsize]
totsize = op.windowsize
chunkspecs = (totsize,DAE.get_chunkspec.(op.inars)..., DAE.get_chunkspec.(op.outspecs,op.f.outtype)...)

ts = first(chunkspecs)
cs = Base.tail(chunkspecs)
inv(DAE.time_per_array(cs[1],[90,90,184],ts)/DAE.time_per_array(cs[1],[1440,720,1],ts))


lonr = range(100,1440,length=100)
timr = range(100,1840,length=100)

getwindows(l,t) = [l,max(l,720.0),t]


times = [DAE.compute_time(getwindows(l,t),chunkspecs) for l in lonr, t in timr]
using Plots
contour(lonr,timr,log.(times))


DAE.compute_time([2.0,2.0,2.0],chunkspecs)

optprob = OptimizationFunction(compute_time, Optimization.AutoForwardDiff(), cons = all_constraints!)
prob = OptimizationProblem(optprob, x0, chunkspecs, lcons = lb, ucons = ub)
sol = solve(prob, OptimizationOptimJL.IPNewton())


lr = DAE.optimize_loopranges(op,5e8)




    @time rechunk_diskarray(aout,a)


# using Plots
# heatmap(aout[:,:,1600])