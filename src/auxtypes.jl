struct ProductArray{T,N,S<:Tuple} <:AbstractArray{T,N}
    members::S
end
function ProductArray(members::Tuple)
    et = Tuple{eltype.(members)...}
    ProductArray{et,length(members),typeof(members)}(members)
end
Base.size(a::ProductArray,i::Int) = length(a.members[i])
Base.size(a::ProductArray) = length.(a.members)
Base.IndexStyle(::Type{<:ProductArray})=Base.IndexCartesian()
Base.getindex(a::ProductArray{<:Any,N}, i::Vararg{Int, N}) where N = getindex.(a.members,i)

using Distributed

function pmap_with_data(f, p::AbstractWorkerPool, c...; initfunc, progress=nothing, kwargs...)
    d = Dict(ip=>remotecall(initfunc, ip) for ip in workers(p))
    allrefs = @spawn d
    function fnew(args...,)
        refdict = fetch(allrefs)
        myargs = fetch(refdict[myid()])
        f(args..., myargs)
    end
    if progress !==nothing
        progress_pmap(fnew,p,c...;progress=progress,kwargs...)
    else
        pmap(fnew,p,c...;kwargs...)
    end
end
pmap_with_data(f,c...;initfunc,kwargs...) = pmap_with_data(f,default_worker_pool(),c...;initfunc,kwargs...) 

"Type used for dispatch to show something is done in input mode"
struct Input end

"Type used for dispatch to show something is done in output mode"
struct Output end