using DiskArrayEngine
import DiskArrayEngine as DAE
using NetCDF, YAXArrays
ds = open_dataset("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc")

#a = NetCDF.open("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc","lccs_class")
#eltype(a)
res = aggregate_diskarray(ds.lccs_class,DAE.KeyConvertDicts.NonHashCountMap,(1=>nothing,2=>nothing,3=>nothing))

res_serial = DAE.compute(res)

using OnlineStats
res = aggregate_diskarray(ds.lccs_class,CountMap{UInt8},(1=>nothing,2=>nothing,3=>nothing))
res_serial = DAE.compute(res)
