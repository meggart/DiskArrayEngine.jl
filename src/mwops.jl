using DiskArrays: DiskArrays, ChunkType, GridChunks, AbstractDiskArray
using Zarr
export InputArray, create_outwindows, GMDWop, create_outars

internal_size(p) = last(last(p))-first(first(p))+1
function steps_per_chunk(p,cs::ChunkType)
    centers = map(x->(first(x)+last(x))/2,p)
    slen = sum(cs) do r
        i1 = searchsortedfirst(centers,first(r))
        i2 = searchsortedlast(centers,last(r))
        length(i1:i2)
    end
    slen/length(cs)
end

"""
Struct specifying the windows of a participating array along each dimension as well as 
the loop axes where this array participates in the loop
"""
struct LoopWindows{W,IL}
    windows::W
    lr::Val{IL}
end


struct InputArray{A,LW<:LoopWindows}
    a::A
    lw::LW
end
InputArray(a::Number;kwargs...) = InputArray(fill(a);kwargs...)
function InputArray(a::AbstractArray;dimsmap = ntuple(identity,ndims(a)),windows = Base.OneTo.(size(a)))
  length(dimsmap) == ndims(a) || throw(ArgumentError("number is dimensions in loop dimension map not equal to ndims(a)"))
  length(windows) == ndims(a) || throw(ArgumentError("number of supplied loop windwos not equal to ndims(a)"))
  lw = LoopWindows(ProductArray(to_window.(windows)),Val((dimsmap...,)))
  InputArray(a,lw)
end

ismem(a::InputArray) = ismem(a.a)
ismem(::AbstractDiskArray) = false
ismem(::Any) = true
getdata(c::InputArray) = c.a
getloopinds(::LoopWindows{<:Any,IL}) where IL = IL 
getsubndims(::LoopWindows{<:Any,IL}) where IL = length(IL)
@inline getloopinds(c) = getloopinds(c.lw)
@inline getsubndims(c) = getsubndims(c.lw)

function create_outwindows(s;dimsmap = ntuple(identity,length(s)),windows = Base.OneTo.(s), chunks = map(_->nothing,s),ismem=false)
  outrp = ProductArray(to_window.(windows))
  (;lw=LoopWindows(outrp,Val((dimsmap...,))),chunks,ismem)
end

mysub(ia,t) = map(li->t[li],getloopinds(ia))

"Returns the full domain that a `DiskArrays.ChunkType` object covers as a unit range"
domain_from_chunktype(ct) = first(first(ct)):last(last(ct))
"Returns the length of a dimension covered by a `DiskArrays.ChunkType` object"
length_from_chunktype(ct) = length(domain_from_chunktype(ct))


"Tests that a supplied list of parent chunks covers the same domain and returns this"
function range_from_parentchunks(pc)
    d = domain_from_chunktype(first(pc))
    for c in pc
        if domain_from_chunktype(c)!=d
            throw(ArgumentError("Supplied parent chunks cover different domains"))
        end
    end
    d
end



function getwindowsize(inars, outspecs)
    d = Dict{Int,Int}()
    for ia in (inars...,outspecs...)
      addsize!(ia.lw,d)
    end
    imax = maximum(keys(d))
    ntuple(i->d[i],imax)
  end
  function addsize!(ia,d)
    for (s,li) in zip(size(ia.windows),getloopinds(ia))
      if haskey(d,li)
        if d[li] != s
          @show d
          @show li,s
          error("Inconsistent Loop windows")
        end
      else
        d[li] = s
      end
    end
  end

struct GMDWop{N,I,O,F<:UserOp,SPL}
    inars::I
    outspecs::O
    f::F
    windowsize::NTuple{N,Int}
    lspl::SPL
end
function GMDWop(inars, outspecs, f)
    s = getwindowsize(inars, outspecs)
    lspl = isa(f.f,BlockFunction) ? nothing : get_loopsplitter(length(s),outspecs)
    GMDWop(inars,outspecs, f, s, lspl)
end

function create_outars(op,plan;par_only=false)
  map(plan.output_chunkspecs,op.f.outtype) do outspec,rettype
    chunks = DiskArrays.GridChunks(output_chunks(outspec,plan.lr))
    chunksize = DiskArrays.approx_chunksize(chunks)
    outsize = last.(last.(chunks.chunks))
    retnmtype = Base.nonmissingtype(rettype)
    if sizeof(rettype)*prod(outsize) > 1e8
      zcreate(retnmtype,outsize...,path=tempname(),fill_value=typemin(retnmtype),chunks=chunksize,fill_as_missing=Missing <: rettype)
    else
      if par_only
        a = Array{rettype,length(outsize)}(undef,outsize...)
        c = RemoteChannel()
        put!(c,a)
      else
        Array{rettype,length(outsize)}(undef,outsize...)
      end
    end
  end
end

