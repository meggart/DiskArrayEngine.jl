using OnlineStats


fit_online!(xout,x::Number) = fit!(xout[],x)
fit_online!(xout,::Missing) = nothing
function fit_online!(xout,x)
  if !all(ismissing,x) 
    fit!(xout[],mean(skipmissing(x)))
  end
end