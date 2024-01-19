using DiskArrayEngine
import DiskArrayEngine as DAE
using OnlineStats
using Zarr, NetCDF
using DataStructures: OrderedDict
using YAXArrays
using DistributedArrays: DArray
using DataStructures: LittleDict
using Cthulhu
mycatchargs = []
ds = open_dataset("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc")
a = view(ds.lccs_class.data,:,:,1)
#a = NetCDF.open("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc","lccs_class")
#eltype(a)
res = aggregate_diskarray(a,DAE.KeyConvertDicts.NonHashCountMap,(1=>nothing,2=>nothing))
res2 = aggregate_diskarray(a,CountMap{UInt8},(1=>nothing,2=>nothing,3=>nothing))

res_serial = DAE.compute(res,threaded=false)


filter(i->last(i)>0,pairs(res_serial[1]))

using Distributed
addprocs(8)
@everywhere using DiskArrayEngine, OnlineStats, NetCDF, YAXArrays
@everywhere begin
    
end



stat = KCDT(UInt8,Int,plusone,minusone,256)
res = aggregate_diskarray(a,CountMap{UInt8,stat},(1=>nothing,2=>nothing,3=>nothing))
@time counts = compute(res,runner=DAE.DaggerRunner,threaded=false)

rmprocs(workers())

# @everywhere begin
#     using LoggingExtras
#     mylogger = EarlyFilteredLogger(FileLogger("$(myid()).log")) do log
#         (log._module == DiskArrayEngine && log.level >= Logging.Debug) || log.level >=Logging.Info
#     end
#     global_logger(mylogger)  
# end

# @time unflushed_buffers = run(r)
#@time res_dagger = DAE.compute(res,threaded=false,runner=DAE.DaggerRunner)
