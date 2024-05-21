using ArchGDAL, GDAL
using DiskArrayEngine
const AG = ArchGDAL
const DAE = DiskArrayEngine
using DiskArrays
using Statistics: median, median!

ds = AG.readraster("/home/fgans/data/geospatial-python-raster-dataset/S2A_31UFU_20200321_0_L2A/TCI.tif")
colorview(RGB,reinterpret(N0f8,permutedims(ds,(3,1,2))[:,:,:]))
windowsize = 7
hw = windowsize÷2
sx,sy = size(ds,1),size(ds,2)
windowsx = [max(1,i-hw):min(sx,i+hw) for i in 1:sx]
windowsy = [max(1,i-hw):min(sy,i+hw) for i in 1:sy]
input = DAE.InputArray(ds,windows = (windowsx,windowsy,1:3))
outspecs = DAE.create_outwindows(size(ds))
f = DAE.create_userfunction(x->round(UInt8,median(x)), UInt8)
op = DAE.GMDWop((input,),(outspecs,),f)
max_memcache = 5e8 #500 MB max memory per worker
lr = DAE.optimize_loopranges(op,max_memcache,x0=[1000.0,1000.0,2.0])

runner = DAE.LocalRunner(op,lr,(zeros(UInt8,size(ds)),))

run(runner)

using Images, FixedPointNumbers
r = only(runner.outars)

a = PermutedDimsArray(reinterpret(N0f8,only(runner.outars)),(3,1,2))
hcat(colorview(RGB,a),colorview(RGB,reinterpret(N0f8,permutedims(ds,(3,1,2))[:,:,:])))
