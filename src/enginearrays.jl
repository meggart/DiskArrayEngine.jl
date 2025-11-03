using DiskArrays
using OnlineStats: OnlineStats
using Statistics
export engine, compute, compute!
abstract type AbstractEngineArray{T,N} <: AbstractDiskArray{T,N} end
struct EngineArray{T,N,P} <: AbstractEngineArray{T,N}
    parent::P
    bcdims::Base.RefValue{NTuple{N,Int}}
end
EngineArray(p) = EngineArray{eltype(p),ndims(p),typeof(p)}(p,Ref(ntuple(identity,ndims(p))))
unengine(a::EngineArray) = a.parent
unengine(a) = a
engine(p;bcdims=ntuple(identity,ndims(p))) = EngineArray{eltype(p),ndims(p),typeof(p)}(p,Ref(bcdims))
engine(p::EngineArray;bcdims=ntuple(identity,ndims(p))) = EngineArray{eltype(p),ndims(p),typeof(p)}(p.parent,bcdims)
function engine(p::AbstractEngineArray;bcdims=nothing)
    if bcdims === nothing
        p
    else
        bcdims=ntuple(identity,ndims(p))
        EngineArray{eltype(p),ndims(p),typeof(p)}(p,Ref(bcdims))
    end
end
Base.parent(p::EngineArray) = p.parent
DiskArrays.readblock!(a::EngineArray,xout,r::OrdinalRange...) = DiskArrays.readblock!(a.parent,xout,r...)
DiskArrays.writeblock!(a::EngineArray,xout,r::OrdinalRange...) = DiskArrays.writeblock!(a.parent,xout,r...)
Base.size(a::EngineArray) = size(a.parent)
DiskArrays.eachchunk(a::EngineArray) = DiskArrays.eachchunk(a.parent)
bcdims(a::EngineArray) = a.bcdims[]
bcdims(p) = ntuple(identity,ndims(p))

function collect_bcdims(A)
    o = Dict{Int,Int}()
    for a in A
        for (i,d) in enumerate(bcdims(a))
            newsize = size(a,i)
            if haskey(o,d)
                oldsize = o[d]
                mergedsize = if newsize == oldsize
                    newsize
                elseif newsize == 1
                    oldsize
                elseif oldsize == 1
                    newsize
                else
                    error("Dimension length do not match $newsize and $oldsize")
                end
                o[d] = mergedsize
            else
                o[d] = size(a,i)
            end
        end
    end
    oc = collect(pairs(o))
    sort!(oc,by=first)
    last.(oc), first.(oc)
end

function Base.mapreduce(f, op, A::AbstractEngineArray...; dims=:, init = nothing, fin = identity)
    s, bcd = collect_bcdims(A)
    nd = maximum(bcd)
    ia = map(A) do ar
        lw = map(bcdims(ar),size(ar)) do idim, sa
            sloop = s[idim]
            if sa == sloop
                1:sa
            else
                @assert sa == 1
                fill(1,sloop)
            end
        end
        InputArray(unengine,windows = lw, dimsmap = bcdims(ar))
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
    windows = map(ntuple(identity,nd)) do idim
        if idim in dims
            Repeated(s[idim])
        else
            1:s[idim]
        end
    end
    outsize = length.(windows)
    # outar = zeros(top,outsize)
    outwindows = (create_outwindows(outsize,windows = windows),)
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
    $f(a::AbstractEngineArray;dims=:,skipmissing=false) = $f(identity,a;dims,skipmissing)
    function $f(ff::Base.Callable,a::AbstractEngineArray;dims=:,skipmissing=false)
        red = skipmissing ? missred($red) : $red
        init = skipmissing ? missing : $i
        mapreduce(ff,red,a;dims,init=$(i)(a))
    end 
    
    
end)
end

Base.mapslices(f,A::AbstractEngineArray...;dims,outchunks=nothing) = mapslices_engine(f,A;dims,outchunks)

function mapslices_engine(f,A...;dims,outchunks=nothing)
    s, bcd = collect_bcdims(A)
    nd = maximum(bcd)
    ia = map(A) do ar
        lw = map(bcdims(ar),size(ar)) do idim, sa
            sloop = s[idim]
            if idim in dims    
                [1:sa]
            elseif sa == sloop
                1:sa
            else
                @assert sa == 1
                Repeated(1,sloop)
            end
        end
        InputArray(unengine(ar),windows = lw, dimsmap = bcdims(ar))
    end
    inputtypes = map(ia) do inar
        if isempty(inar.lw.windows.members)
            eltype(inar.a)
        else
            ndi = sum(i->isa(i,Tuple),inar.lw.windows.members)
            ndi > 0 ? Array{eltype(inar.a),ndi} : eltype(inar.a)
        end
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

Statistics.median(a::AbstractEngineArray;dims=()) = mapslices_engine(median,a;dims)


struct EngineStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
using DiskArrays: ChunkStyle
using Base.Broadcast: DefaultArrayStyle

Base.BroadcastStyle(::Type{<:AbstractEngineArray{<:Any,N}}) where N = EngineStyle{N}()
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
    mapslices_engine(bc.f,bc.args...,dims=())
end
function Base.copyto!(dest::AbstractArray, bc::Base.Broadcast.Broadcasted{EngineStyle{N}}) where {N}
    bcf = Base.Broadcast.flatten(bc)
    size(bcf) == size(dest) || throw(ArgumentError("dest and broadcast expression must have the same size"))
    r = mapslices_engine(bcf.f,bcf.args...,dims=(),outchunks=DiskArrays.approx_chunksize(DiskArrays.eachchunk(dest)))
    compute!(dest,r)
    dest
end
