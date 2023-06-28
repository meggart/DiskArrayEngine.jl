using DiskArrayEngine, Statistics


using Zarr
a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/",fill_as_missing=true)

ae = engine(a)

med = mean(ae,dims=3,skipmissing=true)
med[1000,500,1]
med = mapslices(median ∘ skipmissing,view(ae,:,:,300:1000),dims=3)

med[1000,500,1]



r = minimum(ae,dims=3,skipmissing=true)

s = sum(ae,dims=(1,2),skipmissing=true)

compute(s)

@time r[1000,720]


@time aa = compute(r)



ae2 = ae .+ ae

output = zcreate(Float32,size(ae)...,path=tempname())

ae2[1000,500,100]

ae[1000,500,100]*2

using Plots
heatmap(reverse(permutedims(aa),dims=1))


using Flatten

ae2 = reshape(ae,size(ae)...,1,1)

r[8]
ae2.parent