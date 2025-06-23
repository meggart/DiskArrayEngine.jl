using DiskArrays: DiskArrays, ChunkVector, GridChunks, AbstractDiskArray
using Zarr
export InputArray, create_outwindows, GMDWop, create_outars

internal_size(p) = last(last(p))-first(first(p))+1
function steps_per_chunk(p,cs::ChunkVector)
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
function Base.show(io::IO,::MIME"text/plain",lw::LoopWindows)
  print(io, "Loop Window object with the following ID-Window mapping")
  foreach(getloopinds(lw),lw.windows.members) do li,w
    print(io,"]\n")
    print(io,"$li: Window of length $(length(w)) [")
    showifthere(io,w,1,sep=false)
    showifthere(io,w,2)
    print(io," ... ")
    showifthere(io,w,max(length(w)-1,3),sep=false)
    showifthere(io,w,max(length(w),4))
    
  end
end
function showifthere(io,w,i;sep=true)
  if length(w)>=i
    sep && print(io,", ")
    print(io,w[i])
  end
end
function purify_window(lw::LoopWindows)
  pw = purify_window.(lw.windows.members)
  LoopWindows(ProductArray(pw),lw.lr)
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
  if sort(collect(dimsmap)) != 1:length(dimsmap)
    throw(ArgumentError("Vanishing dimensions in outputs are not allowed. Please add length-1 dummy dimensions for every input axis."))
  end
  (;lw=LoopWindows(outrp,Val((dimsmap...,))),chunks,ismem)
end

mysub(ia,t) = map(li->t[li],getloopinds(ia))

"Returns the full domain that a `DiskArrays.ChunkVector` object covers as a unit range"
domain_from_chunktype(ct) = first(first(ct)):last(last(ct))
"Returns the length of a dimension covered by a `DiskArrays.ChunkVector` object"
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


"""
    GMDWop{N,I,O,F<:UserOp,SPL}

A struct representing a generalized multi-dimensional windowed operation.

# Fields
- `inars::I`: Tuple or collection of input arrays, each wrapped as `InputArray`.
- `outspecs::O`: Tuple or collection of output array specifications.
- `f::F`: The user-supplied operation, typically a UserFunc returned by `create_userfunction`.
- `windowsize::NTuple{N,Int}`: The number of windows along each loop dimension.
- `lspl::SPL`: Loop splitter object or `nothing`, used for parallelization or block processing.

# Description
`GMDWop` encapsulates all information required to perform a lazy windowed operation over multiple input arrays and produce one or more outputs. 
It manages the mapping between logical loop dimensions and physical array dimensions, window sizes, and any loop splitting for efficient computation.

Construct using `GMDWop(inars, outspecs, f)`, where `inars` and `outspecs` are collections of input and output specifications, and `f` is the user operation.
"""
struct GMDWop{N,I,O,F<:UserOp,SPL}
    inars::I
    outspecs::O
    f::F
    windowsize::NTuple{N,Int}
    lspl::SPL
end

function GMDWop(inars, outspecs, f)
  s = getwindowsize(inars, outspecs)
  lspl = isa(f.f, BlockFunction) ? nothing : get_loopsplitter(length(s), outspecs)
  nd = length(s)
  foreach(outspecs) do spec
    li = getloopinds(spec.lw)
    sort(collect(li)) == 1:nd || throw(ArgumentError("Vanishing Dimensions are not allowed for output arrays."))
  end
  GMDWop(inars, outspecs, f, s, lspl)
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

