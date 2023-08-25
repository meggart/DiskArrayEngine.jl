using DiskArrayEngine
using Zarr

using Distributed
addprocs(4,exeflags=["--project=$(@__DIR__)"])
@everywhere begin
    using DiskArrayEngine, Zarr,LoggingExtras

    mylogger = EarlyFilteredLogger(FileLogger("$(myid()).log")) do log
        (log._module == DiskArrayEngine && log.level >= Logging.Debug) || log.level >=Logging.Info
    end
    global_logger(mylogger)  
end


a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/gross_primary_productivity/", fill_as_missing=true);

isdir("rechunked.zarr") && rm("./rechunked.zarr",recursive=true)
aout = zcreate(eltype(a),size(a)...,path="./rechunked.zarr",chunks = (1440,720,1),fill_value=-1.32f0,fill_as_missing=true)

@time rechunk_diskarray(aout,a)


using Plots
heatmap(aout[:,:,1600])