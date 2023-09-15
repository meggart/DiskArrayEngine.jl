export disk_onlinestat
using OnlineStats: OnlineStats
function fit_online!(xout,x,f=identity)
    fx = f(x)
    ismissing(fx) || OnlineStats.fit!(xout[],fx)
end
fin_online(x) = OnlineStats.nobs(x) == 0 ? missing : OnlineStats.value(x);
disk_onlinestat(s::Type{<:OnlineStats.OnlineStat},preproc=identity) = create_userfunction(
    fit_online!,
    Union{Float64,Missing},
    is_mutating = true,
    red = OnlineStats.merge!, 
    init = s, 
    finalize=fin_online,
    buftype = typeof(s()),  
    args = (preproc,)
)

disk_onlinestat(s,preproc=identity) = disk_onlinestat(func_to_online(s),preproc)
func_to_online(::typeof(mean)) = OnlineStats.Mean
func_to_online(::typeof(sum)) = OnlineStats.Sum