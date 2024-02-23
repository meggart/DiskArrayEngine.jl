export disk_onlinestat
using OnlineStats: OnlineStats
function fit_online!(xout,x,f=identity)
    fx = f(x)
    ismissing(fx) || OnlineStats.fit!(xout[],fx)
end
fin_online(x) = OnlineStats.nobs(x) == 0 ? missing : OnlineStats.value(x);
disk_onlinestat(s::Type{<:OnlineStats.OnlineStat},rt=typeof(OnlineStats.value(s())),preproc=identity) = create_userfunction(
    fit_online!,
    rt,
    is_mutating = true,
    red = OnlineStats.merge!, 
    init = s, 
    finalize=fin_online,
    buftype = typeof(s()),
    allow_threads=false,
    args = (preproc,)
)

disk_onlinestat(s,preproc=identity) = disk_onlinestat(func_to_online[s]...,preproc)
func_to_online = Dict([
    mean => (OnlineStats.Mean,Union{Float64,Missing}),
    sum => (OnlineStats.Sum,Union{Float64,Missing}),
    extrema => (OnlineStats.Extrema, Union{Tuple{Float64,Float64},Missing}),
])
