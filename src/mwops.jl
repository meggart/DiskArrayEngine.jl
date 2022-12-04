using DiskArrays: DiskArrays, ChunkType, GridChunks

struct ProcessingSteps{S,T<:AbstractVector{S}} <: AbstractVector{S}
    offset::Int
    v::T
end
Base.size(p::ProcessingSteps,i...) = size(p.v,i...)
Base.getindex(p::ProcessingSteps,i::Int) = p.v[i] .- p.offset

internal_size(p) = last(last(p))-first(first(p))+1
function subset_step_to_chunks(p::ProcessingSteps,cs::ChunkType)
    centers = map(x->(first(x)+last(x))/2,p)
    map(cs) do r
        i1 = searchsortedfirst(centers,first(r))
        i2 = searchsortedlast(centers,last(r))
        ProcessingSteps(0,view(p.v,i1:i2))
    end
end

struct InputArray{A,W,IL}
    a::A
    windows::W
    lr::Val{IL}
end

getdata(c::InputArray) = c.a
getloopinds(::InputArray{<:Any,<:Any,IL}) where IL = IL 


"""
    struct MWOp

A type holding information about the sliding window operation to be done over an existing dimension. 
Field names:

* `rtot` unit range denoting the full range of the operation
* `parentchunks` list of chunk structures of the parent arrays
* `w` size of the moving window, length-2 tuple with steps before and after center
* `steps` range denoting the center coordinates for each step of the op
* `outputs` ids of related outputs and indices of their dimension index
"""
struct MWOp{G<:ChunkType,P}
    rtot::UnitRange{Int64}
    parentchunks::G
    steps::P
    is_ordered::Bool
end
function MWOp(parentchunks; r = first(first(parentchunks)):last(last(parentchunks)), steps=ProcessingSteps(0,r),is_ordered=false)
    MWOp(r, parentchunks, steps, is_ordered)
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

struct MutatingFunction{F}
    f::F
end
struct NonMutatingFunction{F}
    f::F
end

struct UserOp{F,R,I,FILT,A,KW}
    f::F
    red::R
    init::I
    filters::FILT
    args::A
    kwargs::KW
end


applyfilter(f::UserOp,myinwork) = map(docheck, f.filters, myinwork)
apply_function(f::UserOp{<:MutatingFunction},xout,xin) = f.f(xout...,xin...,f.args...;f.kwargs...)
function apply_function(f::UserOp{<:NonMutatingFunction,Nothing},xout,xin)
    r = f.f.f(xin...,f.args...;f.kwargs...)
    if length(xout) == 1
        first(xout) .= r
    else
        foreach(xout,r) do x,y
            x.=y
        end
    end
end
function apply_function(f::UserOp{<:NonMutatingFunction,<:Base.Callable},xout,xin)
    r = f.f.f(xin...,f.args...;f.kwargs...)
    if length(xout) == 1
        first(xout)[] = f.red(first(xout)[],r)
    else
        rr = f.red(xout,r)
        foreach(xout,rr) do x,y
            x.=y
        end
    end
end

struct GMDWop
    inars
    outars
    loopranges
    f::UserOp
end


abstract type Emitter end
struct DirectEmitter end


abstract type Aggregator end

struct ReduceAggregator{F}
    op::F
end



