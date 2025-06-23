export disk_onlinestat
using OnlineStats: OnlineStats
function fit_online!(xout,x,f=identity,fg=identity)
    fx = f(x)
    ismissing(fx) || OnlineStats.fit!(xout[],fx)
end
function fit_online!(xout,x,group,f,fg)
    ind = fg(group)
    ismissing(ind) && return
    stat = xout[ind]
    fx = f(x)
    ismissing(fx) || isnan(fx) || OnlineStats.fit!(stat,fx)
end

fin_online(x) = OnlineStats.nobs(x) == 0 ? missing : OnlineStats.value(x);
disk_onlinestat(s::Type{<:OnlineStats.OnlineStat},rt=typeof(OnlineStats.value(s())),preproc=identity,groupconv=identity) = create_userfunction(
    fit_online!,
    rt,
    is_mutating = true,
    red = OnlineStats.merge!, 
    init = s, 
    finalize=fin_online,
    buftype = typeof(s()),
    allow_threads=false,
    args = (preproc,groupconv)
)

struct DerivedOnlineStat{P,V,F} <: OnlineStats.OnlineStat{Number}
    parent::P
    valuefunc::V
    fitfunc::F
end
DerivedOnlineStat{P,V,F}() where {P,V,F} = DerivedOnlineStat(P(),V,F)
OnlineStats.fit!(s::DerivedOnlineStat,x::Number) = s.fitfunc(s.parent,x)
OnlineStats.value(s::DerivedOnlineStat) = s.valuefunc(s.parent)
OnlineStats.nobs(s::DerivedOnlineStat) = OnlineStats.nobs(s.parent)


disk_onlinestat(s,args...) = disk_onlinestat(func_to_online[s]...,args...)
has_onlineversion(f) = f in keys(func_to_online)

const func_to_online = Dict([
    mean => (OnlineStats.Mean,Union{Float64,Missing}),
    sum => (OnlineStats.Sum,Union{Float64,Missing}),
    extrema => (OnlineStats.Extrema, Union{Tuple{Float64,Float64},Missing}),
    maximum => (DerivedOnlineStat{OnlineStats.Extrema,v->last(OnlineStats.value(v)),OnlineStats.fit!}, Union{Float64,Missing}),
    minimum => (DerivedOnlineStat{OnlineStats.Extrema,v->first(OnlineStats.value(v)),OnlineStats.fit!}, Union{Float64,Missing}),
])
