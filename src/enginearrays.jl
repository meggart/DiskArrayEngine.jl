using DiskArrays
using OnlineStats: OnlineStats
using Statistics
export engine, compute, compute!

struct EngineArray{T,N,P} <: AbstractDiskArray{T,N}
    parent::P
    bcdims::Base.RefValue{NTuple{N,Int}}
end
EngineArray(p) = EngineArray{eltype(p),ndims(p),typeof(p)}(p,Ref(ntuple(identity,ndims(p))))
engine(p;bcdims=ntuple(identity,ndims(p))) = EngineArray{eltype(p),ndims(p),typeof(p)}(p,Ref(bcdims))
Base.parent(p::EngineArray) = p.parent
DiskArrays.readblock!(a::EngineArray,xout,r::OrdinalRange...) = DiskArrays.readblock!(a.parent,xout,r...)
DiskArrays.writeblock!(a::EngineArray,xout,r::OrdinalRange...) = DiskArrays.writeblock!(a.parent,xout,r...)
Base.size(a::EngineArray) = size(a.parent)
DiskArrays.eachchunk(a::EngineArray) = DiskArrays.eachchunk(a.parent)

function collect_bcdims(A)
    o = Set{Tuple{Int,Int}}()
    for a in A
        for (i,d) in enumerate(a.bcdims[])
            push!(o,(size(a,i),d))
        end
    end
    oc = collect(o)
    allunique(first.(oc)) || error("Lengths don't match")
    sort!(oc,by=last)
    first.(oc), last.(oc)
end

function Base.mapreduce(f, op, A::EngineArray...; dims=:, init = nothing, fin = identity)
    s, bcd = collect_bcdims(A)
    nd = maximum(bcd)
    ia = map(A) do ar
        InputArray(ar.parent,dimsmap = ar.bcdims[])
    end
    # tf = Base.promote_op(f,Base.nonmissingtype.(eltype.(A))...)
    # top = Base.promote_op(op,tf,tf)
    # if any(a->eltype(a) >: Missing,A) 
    #     top = Union{top,Missing}
    # end
    tf = Base.promote_op(f,eltype.(A)...)
    top = Base.promote_op(op,tf,tf)
    func = create_userfunction(f,top,red=op,init=init,buftype=top,finalize=fin)
    if dims === Colon()
        dims = ntuple(identity,nd)
    end
    outdims = setdiff(ntuple(identity,nd),dims)
    outsize = (s[outdims]...,)
    # outar = zeros(top,outsize)
    outwindows = (create_outwindows(outsize,dimsmap=(outdims...,)),)
    op = GMDWop(ia, outwindows, func)
    first(results_as_diskarrays(op))
end
missred(f) = (a,b)-> ismissing(a) ? b : ismissing(b) ? a : f(a,b)
init_min(a) = typemax(Base.nonmissingtype(eltype(a)))
init_max(a) = typemin(Base.nonmissingtype(eltype(a)))
init_sum(::AbstractArray{<:Union{AbstractFloat,Missing}}) = zero(Float64)
init_sum(::AbstractArray{<:Union{Unsigned,Missing}}) = zero(UInt64)
init_sum(::AbstractArray{<:Union{Signed,Missing}}) = zero(Int64)
init_sum(a) = zero(eltype(a))
init_ex(a) =  typemax(Base.nonmissingtype(eltype(a))),typemin(Base.nonmissingtype(eltype(a)))

extrred(x,y) = min(first(x),first(y)),max(last(x),last(y))

wrap_reduction(a) = a
wrap_reduction(a::OnlineStats.OnlineStat) = OnlineStats.value(a)

for (f,red,i) in (
    (:(Base.maximum),:max,:init_max),
    (:(Base.minimum),:min,:init_min),
    (:(Base.extrema),:extrred,:init_ex),
    (:(Base.sum),:(+),:init_sum),
)
    eval(quote
        $f(a::EngineArray;dims=:,skipmissing=false) = $f(identity,a;dims,skipmissing)
        function $f(ff::Base.Callable,a::EngineArray;dims=:,skipmissing=false)
            red = skipmissing ? missred($red) : $red
            init = skipmissing ? missing : $i
            mapreduce(ff,red,a;dims,init=$(i)(a))
        end 


    end)
end

function Base.mapslices(f,A::EngineArray...;dims,outchunks=nothing)
    s, bcd = collect_bcdims(A)
    nd = maximum(bcd)
    ia = ntuple(length(A)) do ii
        ar = A[ii]
        ms = size(ar)
        windows = Base.OneTo.(ms)
        for d in dims
            idim = findfirst(==(d),ar.bcdims[])
            if idim !== nothing
                windows = Base.setindex(windows,(windows[idim],),idim)
            end
        end
        InputArray(ar.parent,dimsmap = ar.bcdims[],windows=windows)
    end
    inputtypes = map(ia) do inar
        ndi = sum(i->isa(i,Tuple),inar.lw.windows.members)
        ndi > 0 ? Array{eltype(inar.a),ndi} : eltype(inar.a)
    end

    tf = get_infered_types(f,inputtypes)
    func = create_userfunction(f,tf,buftype=tf)
    outdims = ntuple(identity,nd)
    outsize = copy(s)
    for d in dims
        outsize[d] = 1
    end
    if outchunks === nothing
        outchunks = ntuple(_->nothing,length(outsize))
    end
    outwindows = (create_outwindows((outsize...,),dimsmap=(outdims...,),chunks=outchunks),)
    op = GMDWop(ia, outwindows, func)
    first(results_as_diskarrays(op))
end



get_infered_types(f,inputtypes) = Base.promote_op(f,inputtypes...)

Statistics.median(a::EngineArray;dims=()) = mapslices(median,a;dims)

function compute!(ret,a::DiskArrayEngine.GMWOPResult)
    lr = DiskArrayEngine.optimize_loopranges(a.op,5e8,tol_low=0.2,tol_high=0.05,max_order=2)
    outars = create_outars(a.op,lr)
    iout = findfirst(i->Val(i)===a.ires,1:length(outars))
    if ret !== nothing
        outars = Base.setindex(outars,ret,iout)
    end
    r = DiskArrayEngine.LocalRunner(a.op,lr,outars,threaded=true)
    run(r)
    outars[iout]
end
function compute(a::DiskArrayEngine.GMWOPResult)
    compute!(nothing,a)
end

struct EngineStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
using DiskArrays: ChunkStyle
using Base.Broadcast: DefaultArrayStyle

Base.BroadcastStyle(::EngineStyle{N}, ::EngineStyle{M}) where {N,M} = EngineStyle{max(N, M)}()
function Base.BroadcastStyle(::EngineStyle{N}, ::DefaultArrayStyle{M}) where {N,M}
    return EngineStyle{max(N, M)}()
end
function Base.BroadcastStyle(::DefaultArrayStyle{M}, ::EngineStyle{N}) where {N,M}
    return EngineStyle{max(N, M)}()
end
function Base.BroadcastStyle(::EngineStyle{N}, ::ChunkStyle{M}) where {N,M}
    return EngineStyle{max(N, M)}()
end
function Base.BroadcastStyle(::ChunkStyle{M}, ::EngineStyle{N}) where {N,M}
    return EngineStyle{max(N, M)}()
end

function Base.copy(bc::Base.Broadcast.Broadcasted{EngineStyle{N}}) where {N}
    bc = Base.Broadcast.flatten(bc)
    mapslices(bc.f,bc.args...,dims=())
end
function Base.copyto!(dest::AbstractArray, bc::Base.Broadcast.Broadcasted{EngineStyle{N}}) where {N}
    bcf = Base.Broadcast.flatten(bc)
    size(bcf) == size(dest) || throw(ArgumentError("dest and broadcast expression must have the same size"))
    r = mapslices(bcf.f,bcf.args...,dims=(),outchunks=DiskArrays.approx_chunksize(DiskArrays.eachchunk(dest)))
    compute!(dest,r)
    dest
end