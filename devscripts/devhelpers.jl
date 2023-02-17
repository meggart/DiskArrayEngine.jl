using OnlineStats


fit_online!(xout,x::Number) = fit!(xout[],x)
fit_online!(xout,::Missing) = nothing
