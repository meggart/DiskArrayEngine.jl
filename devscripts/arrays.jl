using DiskArrayEngine
using DiskArrays

struct EngineArray{T,N,P} <: AbstractDiskArray{T,N}
    parent::P
    bcdims::Base.RefValue{NTuple{N,Int}}
end
EngineArray(p) = EngineArray{eltype(p),ndims(p),typeof(p)}(p,Ref(ntuple(identity,ndims(p))))
Base.parent(p::EngineArray) = p.parent
DiskArrays.readblock!(a::EngineArray,xout,r::OrdinalRange...) = DiskArrays.readblock!(a.parent,xout,r...)
DiskArrays.writeblock!(a::EngineArray,xout,r::OrdinalRange...) = DiskArrays.writeblock!(a.parent,xout,r...)
Base.size(a::EngineArray) = size(a.parent)
DiskArrays.eachchunk(a::EngineArray) = DiskArrays.eachchunk(a.parent)

using NetCDF
a = NetCDF.open("../../../Documents/data/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc","lccs_class")

ae = EngineArray(a)

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

function Base.mapreduce(f, op, A::EngineArray...; dims=:, init = nothing)
    s, bcd = collect_bcdims(A)
    nd = maximum(bcd)
    ia = map(A) do ar
        InputArray(ar.parent,dimsmap = ar.bcdims[])
    end
    tf = Core.Compiler.return_type(f,Tuple{eltype.(A)...})

    top = Core.Compiler.return_type(op,Tuple{tf})
    func = create_userfunction(f,top,red=op,init=init,buftype=tf)
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
using Statistics
Base.maximum(a::EngineArray;dims=:) = maximum(identity,a;dims)
Base.maximum(f::Base.Callable,a::EngineArray)
r = mapreduce(identity,min,ae,init=0xff,dims=1)

r[32000,1]