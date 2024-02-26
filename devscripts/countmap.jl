using DiskArrayEngine
import DiskArrayEngine as DAE
using NetCDF, YAXArrays
ds = open_dataset("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc")

#a = NetCDF.open("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc","lccs_class")
#eltype(a)
res = aggregate_diskarray(ds.lccs_class.data,DAE.KeyConvertDicts.NonHashCountMap,(1=>nothing,2=>nothing,3=>nothing))
res_serial = DAE.compute(res)

using OnlineStats
res = aggregate_diskarray(ds.lccs_class.data,CountMap{UInt8},(1=>nothing,2=>nothing,3=>nothing))
res_serial = DAE.compute(res)

using Distributed
addprocs(16)
@everywhere begin
    using Pkg; Pkg.activate("./devscripts")
    using DiskArrayEngine, NetCDF, YAXArrays
end
res = aggregate_diskarray(ds.lccs_class.data,DAE.KeyConvertDicts.NonHashCountMap,(1=>nothing,2=>nothing,3=>nothing))
@time res_par = DAE.compute(res,runner=DAE.DaggerRunner)

