using DiskArrayEngine
import DiskArrayEngine as DAE
using Zarr, NetCDF
import PyramidScheme as PS

lcfile = NetCDF.open("/home/fgans/data/LC_CCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc")
lc = view(lcfile["lccs_class"],:,:,1)

n_level = PS.compute_nlevels(lc)

dim = (lcfile["lon"][:],lcfile["lat"][:])
pyramid_sizes =  [ceil.(Int, size(lc) ./ 2^i) for i in 1:n_level]
pyramid_axes = [PS.agg_axis.(dim,2^i) for i in 1:n_level]

outarrs = PS.output_arrays(pyramid_sizes, UInt8);



"""
    ESALCMode(counts)

"""
struct ESALCMode
    counts::Channel{Vector{Int}}
end
function ESALCMode() 
    ch = Channel{Vector{Int}}(Threads.nthreads())
    for _ in 1:Threads.nthreads()
        put!(ch,zeros(Int,256))
    end
    ESALCMode(ch)
end
function (f::ESALCMode)(x)
    cv = take!(f.counts)
    fill!(cv,0)
    for ix in x
        cv[Int(ix)+1] += 1
    end
    _,ind = findmax(cv)
    put!(f.counts,cv)
    UInt8(ind-1)
end

m = ESALCMode()

@time PS.fill_pyramids(lc,outarrs,m,true,threaded=false)