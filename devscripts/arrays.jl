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

using Zarr
a = zopen("/home/fgans/data/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/air_temperature_2m/")

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
    tf = Core.Compiler.return_type(f,Tuple{Base.nonmissingtype.(eltype.(A))...})
    top = Core.Compiler.return_type(op,Tuple{tf,tf})
    if any(a->eltype(a) >: Missing,A) 
        top = Union{top,Missing}
    end
    func = create_userfunction(f,top,red=op,init=init,buftype=top)
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
missred(f) = (a,b)-> ismissing(a) ? b : ismissing(b) ? a : f(a,b)
init_min(a) = typemax(Base.nonmissingtype(eltype(a)))
init_max(a) = typemin(Base.nonmissingtype(eltype(a)))
for (f,red,i) in (
    (:(Base.maximum),:max,:init_max),
    (:(Base.minimum),:min,:init_min),
    (:(Base.extrema),)
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


function compute(a::DiskArrayEngine.GMWOPResult)

    ret = zeros(eltype(a),size(a))
    lr = DiskArrayEngine.optimize_loopranges(a.op,1e9,tol_low=0.2,tol_high=0.05,max_order=2)
    outars = ntuple(i->Val(i)==a.ires ? ret : nothing,length(a.op.outspecs))
    r = DiskArrayEngine.LocalRunner(a.op,lr,outars,threaded=true)
    run(r)
    ret
end

r = minimum(ae,dims=3,skipmissing=true)

@time r[1000,720]


@time aa = compute(r)

using Plots
heatmap(reverse(permutedims(aa),dims=1))


