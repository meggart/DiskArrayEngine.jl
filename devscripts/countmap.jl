using DiskArrayEngine
import DiskArrayEngine as DAE
using OnlineStats
using Zarr, NetCDF
using DataStructures: OrderedDict
using YAXArrays
using DistributedArrays: DArray
ds = open_dataset("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc")
a = ds.lccs_class.data
#a = NetCDF.open("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc","lccs_class")
#eltype(a)
f = OnlineStats.CountMap{UInt8}
isa(f,DataType)
res = aggregate_diskarray(a,OnlineStats.CountMap{UInt8},(1=>nothing,2=>nothing,3=>nothing))

#@time res_serial = DAE.compute(res,threaded=false)

using Distributed
addprocs(8)
@everywhere using DiskArrayEngine, OnlineStats, NetCDF, YAXArrays
counts = compute(res,runner=DAE.DaggerRunner,threaded=false)


# @everywhere begin
#     using LoggingExtras
#     mylogger = EarlyFilteredLogger(FileLogger("$(myid()).log")) do log
#         (log._module == DiskArrayEngine && log.level >= Logging.Debug) || log.level >=Logging.Info
#     end
#     global_logger(mylogger)  
# end

# @time unflushed_buffers = run(r)
#@time res_dagger = DAE.compute(res,threaded=false,runner=DAE.DaggerRunner)
