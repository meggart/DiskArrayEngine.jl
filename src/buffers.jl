using FileWatching.Pidfile

abstract type ArrayBuffer end

struct InArrayBuffer{A,O,LW} <: ArrayBuffer
    a::A
    offsets::O
    lw::LW
end
struct OutArrayBuffer{A,O,LW,F} <: ArrayBuffer
    a::A
    offsets::O
    lw::LW
    finalize::F
    nwritten::Base.RefValue{Int64}
    ntot::Int
end

ArrayBuffer(a,offsets,lw) = InArrayBuffer(a,offsets,lw)

getdata(c::ArrayBuffer) = c.a
getloopinds(b::ArrayBuffer) = getloopinds(b.lw)

"Determine the needed buffer size for a given Input array and loop ranges lr"
function getbufsize(ia, lr)
    map(windowbuffersize,mysub(ia,lr.members),ia.lw.windows.members)
end
windowbuffersize(looprange, window) = maximum(c->internal_size(inner_index(window,c)),looprange)

"Creates buffers for input arrays"
function generate_inbuffers(inars,loopranges)
    map(inars) do ia
        et = eltype(ia.a)
        #@show loopranges

        Array{et}(undef,getbufsize(ia,loopranges))
    end
end

is_init_callable(::Any) = Val{false}()
is_init_callable(::Function) = Val{true}()
is_init_callable(::Union{DataType,UnionAll}) = Val{true}()
array_from_init(::Nothing,buftype,bufsize) = zeros(buftype,bufsize)
array_from_init(init,buftype,bufsize) = array_from_init(init,is_init_callable(init),buftype,bufsize)
array_from_init(init,::Val{true},buftype,bufsize) = buftype[init() for _ in CartesianIndices(bufsize)]
array_from_init(init,::Val{false},buftype,bufsize) = buftype[init for _ in CartesianIndices(bufsize)]

#buftype_from_init(_,ia) =

"Create buffer for single output"
function generate_raw_outbuffer(init,buftype,bufsize) 
    array_from_init(init,buftype,bufsize)
end

compute_repeat(w,l,i) = compute_repeat(get_overlap(w),w,l,i)
compute_repeat(::NonOverlapping,_,_,_) = 1
compute_repeat(::Overlapping,_,_,_) = error("Not implemented yet")
function compute_repeat(::Repeating,w,l,i)
    i_current_looprange = findfirst(isequal(i),l)
    w1 = w[first(i)]
    first_window_occurrence = findfirst(==(w1),w)
    n_before = if first_window_occurrence < first(i)
        firstaffectedlooprange = findfirst(i->in(first_window_occurrence,i),l)
        all(==(w1),w[firstaffectedlooprange]) || error("Windows of repeated outputs don't align")
        i_current_looprange - firstaffectedlooprange
    else
        0
    end
    w2 = w[last(i)]
    last_window_occurrence = findlast(==(w2),w)
    n_after = if last_window_occurrence > last(i)
        lastaffectedlooprange = findlast(i->in(last_window_occurrence,i),l)
        all(==(w2),w[lastaffectedlooprange]) || error("Windows of repeated outputs don't align")
        lastaffectedlooprange - i_current_looprange
    else
        0
    end
    return 1+n_before+n_after
end

"""
Compute how often a buffer needs to be passed to the computation before it can be flushed to the
output array
"""
function bufferrepeat(ind,loopranges,lw) 
    # Repeats because a dimension is missing from the loop
    baserep = prod(size(loopranges)) ÷ prod(mysub(lw,size(loopranges)))
    windowmembers = lw.windows.members
    mylr = mysub(lw,loopranges.members)
    myind = mysub(lw,ind)
    @assert length(windowmembers) == length(mylr)
    innerrepeat = map(windowmembers,mylr,myind) do w,l,i
        r = compute_repeat(w,l,i)
        r
    end
    baserep * prod(innerrepeat)
end

"Creates buffers for all outputs, results in a tuple of Dicts holding the collection for each output"
function generate_outbuffers(outars,func,loopranges)
    generate_outbuffer_collection.(outars,func.buftype,(loopranges,),func.finalize)
end

struct BufferIndex{N}
    indranges::NTuple{N,UnitRange{Int}}
end
offset_from_range(r) = first(r) .- 1
offset_from_range(r::BufferIndex) = offset_from_range.(r.indranges)

"""
    struct EmptyInput

A kind of placeholder array generating an artificial input in the DAG graph, which
only gets filled during the computation. Mainly used to reserve an input buffer.  
"""
struct EmptyInput{T,N}
    s::NTuple{N,Int}
end
Base.eltype(::EmptyInput{T}) where T = T
Base.ndims(::EmptyInput{<:Any,N}) where N = N


"Reads data from input array `ia` along domain `r` into `buffer`"
function read_range(r,ia,buffer)
    fill!(buffer,zero(eltype(buffer)))
    inds = get_bufferindices(r,ia)
    if !isa(ia.a,EmptyInput)
        buffer[Base.OneTo.(length.(inds.indranges))...] = ia.a[inds.indranges...]
    end
    ArrayBuffer(buffer,offset_from_range(inds),purify_window(ia.lw))
end

function get_bufferindices(r,outspecs)
    mywindowrange = mysub(outspecs,r)
    BufferIndex(map(outspecs.lw.windows.members,mywindowrange) do w,r
        i = inner_index(w,r)
        first(first(i)):last(last(i))
    end)
end
get_bufferindices(r::BufferIndex,_) = r

struct OutputAggregator{K,V,N,R,F}
    buffers::Dict{K,V}
    bufsize::NTuple{N,Int}
    repeats::R
    finalize::F
end

"""
Removes all outout buffers from an output aggregator that have been successfully put to disk
"""
function clean_aggregator(o::OutputAggregator)
    allkeys = collect(keys(o.buffers))
    for k in allkeys
        if mustdelete(o.buffers[k])
            delete!(o.buffers,k)
        end
    end
end

function merge_outbuffer_collection(o1::OutputAggregator, o2::OutputAggregator,red)
    if o1.bufsize != o2.bufsize
        @info "Warning something is really off with buffer sizes"
    end
    o3 = merge(o1.buffers,o2.buffers) do b1,b2
        n1 = b1.nwritten[]
        n2 = b2.nwritten[]
        @debug myid(), "Merging aggregators of lengths ", n1[], " and ", n2[], " when total mustwrites is ", ntot1
        @assert b1.offsets == b2.offsets
        @assert b1.lw.lr == b2.lw.lr
        @assert length(b1.lw.windows.members) == length(b2.lw.windows.members)
        for (m1,m2) in zip(b1.lw.windows.members,b2.lw.windows.members)
            @assert m1==m2
        end
        @assert b1.ntot == b2.ntot
        @assert b1.finalize == b2.finalize
        merged = red.(b1.a,b2.a)
        if !isa(merged, AbstractArray)
            merged = fill(merged)
        end
        OutArrayBuffer(merged,b1.offsets,b1.lw,b1.finalize,Ref(n1[]+n2[]),b1.ntot)
    end
    OutputAggregator(o3,o1.bufsize,o1.repeats,o1.finalize)
end

buffer_mergefunc(red,_) = (buf1,buf2) -> merge_outbuffer_collection.(buf1,buf2,red)

function merge_all_outbuffers(outbuffers,red)
    @debug "Merging output buffers $(typeof(outbuffers))"
    r = reduce(buffer_mergefunc(red,eltype(outbuffers)),outbuffers)
    @debug "Successfully merged and returning $(typeof(r))"
    r
end

function flush_all_outbuffers(outbuffers,outars,piddir)
    @assert length(outbuffers) == length(outars)
    for (coll,outar) in zip(outbuffers,outars)
        allkeys = collect(keys(coll.buffers))
        @debug "Putting keys $allkeys"
        for k in allkeys
            put_buffer(k,coll.buffers[k], outar, piddir)
        end
        clean_aggregator(coll)
    end
    outbuffers
end
  
function generate_outbuffer_collection(ia,buftype,loopranges,finalize) 
    nd = getsubndims(ia)
    bufsize = getbufsize(ia,loopranges)
    d = Dict{BufferIndex{nd},OutArrayBuffer{Array{buftype,nd},NTuple{nd,Int},typeof(purify_window(ia.lw)),typeof(finalize)}}()
    reps = precompute_bufferrepeat(loopranges,ia)
    OutputAggregator(d,bufsize,reps,finalize)
end

struct ConstDict{V}
    val::V
end
Base.getindex(c::ConstDict,_) = c.val

function precompute_bufferrepeat(lr, outspec)
    r = [get_bufferindices(r,outspec) => bufferrepeat(r,lr,outspec.lw) for r in lr]
    if allequal(last.(r))
        ConstDict(last(first(r)))
    else
        Dict(r)
    end
end

"Extracts or creates output buffer as an ArrayBuffer"
function extract_outbuffer(r,outspecs,init,buftype,buffer::OutputAggregator)
    inds = get_bufferindices(r,outspecs)
    offsets = offset_from_range(inds)
    b = get!(buffer.buffers,inds) do 
        buf = generate_raw_outbuffer(init,buftype,buffer.bufsize)
        ntot = buffer.repeats[inds]
        buf = OutArrayBuffer(buf,offsets,purify_window(outspecs.lw),buffer.finalize,Ref(0),ntot)
    end
    b.nwritten[] = b.nwritten[]+1 
    b
end

mustdelete(buffer::OutArrayBuffer) = buffer.nwritten[] == -1 

"Check if maximum number of aggregations has happened for a buffer"
function mustwrite(buffer::OutArrayBuffer) 
    if buffer.nwritten[] > buffer.ntot
        error("Something is wrong, buffer got wrapped more often than it should. Make sure to use a runner only once")
    else
        buffer.nwritten[] == buffer.ntot
    end
end

extract_channel(x) = x
put_channel(c,x) = nothing
extract_channel(x::RemoteChannel) = first(x)
put_channel(c::RemoteChannel,x) = put!(c,x)

"Checks if output buffers have accumulated to the end and exports to output array"
function put_buffer(r, bufnow, outarc, piddir)
  @debug "Putting buffers"
  bufinds = get_bufferindices(r,bufnow)
  if mustwrite(bufnow)
    fin = bufnow.finalize
    outar = extract_channel(outarc)
    i1 = first.(axes(outar))
    i2 = last.(axes(outar))
    inds = bufinds.indranges
    skip1 = max.(0,i1.-first.(inds))
    skip2 = max.(0,last.(inds).-i2)
    inds2 = range.(first.(inds).+skip1,last.(inds).-skip2)
    r2 = range.(1 .+ skip1, skip1 .+ length.(inds2))
    if piddir !== nothing
        @debug "$(myid()) acquiring lock $piddir to write to $inds2"
        Pidfile.mkpidlock(piddir,wait=true,stale_age=100) do
            broadcast!(fin,view(outar,inds2...),bufnow.a[r2...])
        end
    else
        @debug "$(myid()) Writing data without piddir to $inds2"
        @debug "$outar"
        @debug "$(bufnow.a) $r2"
        broadcast!(fin,view(outar,inds2...),bufnow.a[r2...])
    end
    bufnow.nwritten[] = -1
    put_channel(outarc,outar)
    true
  else
    false
  end
end

"Function to finalize and remove buffers without writing to an array"
function put_buffer(_, bufnow, ::Nothing, _)
  @debug "Putting buffers"
  if mustwrite(bufnow)
    fin = bufnow.finalize
    res = broadcast(fin,bufnow.a)
    bufnow.nwritten[] = -1
    res
  else
    nothing
  end
end

function create_buffers(inars, outars, f, loopranges)
    inbuffers_pure = generate_inbuffers(inars,loopranges)
    outbuffers_pure = generate_outbuffers(outars,f,loopranges)
    inbuffers_pure, outbuffers_pure
end