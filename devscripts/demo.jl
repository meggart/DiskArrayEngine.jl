using Revise, DiskArrayEngine
using Zarr
a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

b = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/min_air_temperature_2m/", fill_as_missing=true)

inputs = InputArray(a),InputArray(b)


outwindows = (create_outwindows(size(a)),)



func = create_userfunction(-,Union{Float32,Missing})

op = GMDWop(inputs, outwindows, func)

r,  = results_as_diskarrays(op)

data = r[1000,500,:]

using Plots
plot(data)
plot(a[1000,500,:].-b[1000,500,:])


heatmap(r[:,:,100])
heatmap(a[:,:,100].-b[:,:,100])




lr = DiskArrayEngine.optimize_loopranges(op,5e8,tol_low=0.2,tol_high=0.05,max_order=2)

outpath = tempname()
c = zzeros(Float32,size(a)...,chunks = (90,90,480),fill_value=NaN32,path=outpath);


r = DiskArrayEngine.LocalRunner(op,lr,(c,),threaded=true)
run(r)

heatmap(c[:,:,100])

#Temporal mean
inputs = (InputArray(a),)
outwindows = (create_outwindows((size(a,1),size(a,2)),
    dimsmap = (1,2),
),)

using OnlineStats
function fit_online!(xout,x)
  ismissing(x) || fit!(xout[],x)
end
init = ()->OnlineStats.Mean()
fin_online(x) = nobs(x) == 0 ? missing : OnlineStats.value(x)
f = create_userfunction(
    fit_online!,
    Float64,
    is_mutating = true,
    red = OnlineStats.merge!, 
    init = init, 
    finalize=fin_online,
    buftype = Mean,  
)

op = GMDWop(inputs, outwindows, f)

r,  = results_as_diskarrays(op)

r[1:10,1:10]

c = zeros(1440,720)
lr = DiskArrayEngine.optimize_loopranges(op,5e8,tol_low=0.2,tol_high=0.05,max_order=2)
r = DiskArrayEngine.LocalRunner(op,lr,(c,),threaded=true)
run(r)

heatmap(c)

# Median
using Statistics

a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/", fill_as_missing=true)

inputs = (InputArray(a,
    windows=(1:1440,1:720,[1:size(a,3)])
),)
outwindows = (create_outwindows(
    (1440,720),
    dimsmap = (1,2),
),)
f = create_userfunction(
    median ∘ skipmissing,
    Float32,   
)

op = GMDWop(inputs, outwindows, f)
r,  = results_as_diskarrays(op)

r[1:10,1:10]


c = zeros(1440,720)
lr = DiskArrayEngine.optimize_loopranges(op,5e8,tol_low=0.2,tol_high=0.05,max_order=2)
r = DiskArrayEngine.LocalRunner(op,lr,(c,),threaded=true)
run(r)

heatmap(c)

# Zonal and monthly means in one operation
using CFTime, StatsBase

t = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/time/", fill_as_missing=true)
t.attrs["units"]
tvec = timedecode(t[:],t.attrs["units"])
years, nts = rle(yearmonth.(tvec))
cums = [0;cumsum(nts)]
stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(nts)]

inputs = (InputArray(a,
    windows=(1:1440,1:720,stepvectime),
),)
outwindows = (create_outwindows(
    (720,480),
    dimsmap = (2,3),
    chunks = (90,24),
),)

function myfunc(x)
  all(ismissing,x) ? (0,zero(eltype(x))) : (1,mean(skipmissing(x)))
end

function reducefunc((n1,s1),(n2,s2))
  (n1+n2,s1+s2)
end
init = ()->(0,zero(Float64))
fin(x) = last(x)/first(x)
f = create_userfunction(
  myfunc,
  Union{Float32,Missing},
  red = reducefunc, 
  init = init, 
  finalize=fin,
  buftype = Tuple{Int,Union{Float32,Missing}},  
)

op = GMDWop(inputs, outwindows, f)
r,  = results_as_diskarrays(op)


plot(r[300,:])

lr = DiskArrayEngine.optimize_loopranges(op,5e8,tol_low=0.2,tol_high=0.05,max_order=2)

outpath = tempname()
c = zzeros(Float32,720,480,chunks = (180,120),fill_value=NaN32,path=outpath);

r = DiskArrayEngine.LocalRunner(op,lr,(c,),threaded=true)
run(r)

heatmap(c[:,:])