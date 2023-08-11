struct ArrayBuffer{A,O,LW}
    a::A
    offsets::O
    lw::LW
end

getdata(c::ArrayBuffer) = c.a
getloopinds(b::ArrayBuffer) = getloopinds(b.lw)

"Determine the needed buffer size for a given Input array and loop ranges lr"
function getbufsize(ia, lr)
    map(mysub(ia,lr.members),ia.lw.windows.members) do cr,op
        maximum(c->internal_size(op[c]),cr)
    end
end

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
    generate_outbuffer_collection.(outars,func.buftype,(loopranges,))
end

struct BufferIndex{N}
    indranges::NTuple{N,UnitRange{Int}}
end
offset_from_range(r) = first(r) .- 1
offset_from_range(r::BufferIndex) = offset_from_range.(r.indranges)

"Reads data from input array `ia` along domain `r` into `buffer`"
function read_range(r,ia,buffer)
    fill!(buffer,zero(eltype(buffer)))
    inds = get_bufferindices(r,ia)
    buffer[Base.OneTo.(length.(inds.indranges))...] = ia.a[inds.indranges...]
    ArrayBuffer(buffer,offset_from_range(inds),ia.lw)
end

function get_bufferindices(r,outspecs)
    mywindowrange = mysub(outspecs,r)
    BufferIndex(map(outspecs.lw.windows.members,mywindowrange) do w,r
        i = w[r]
        first(first(i)):last(last(i))
    end)
end
get_bufferindices(r::BufferIndex,_) = r

struct OutputAggregator{K,V,N}
    buffers::Dict{K,V}
    bufsize::NTuple{N,Int}
end


function merge_outbuffer_collection(o1::OutputAggregator, o2::OutputAggregator,red)
    @assert o1.bufsize == o2.bufsize
    o3 = merge(o1.buffers,o2.buffers) do (n1,ntot1,b1),(n2,ntot2,b2)
        @debug myid(), "Merging aggregators of lengths ", n1[], " and ", n2[], " when total mustwrites is ", ntot1
        @assert b1.offsets == b2.offsets
        @assert b1.lw.lr == b2.lw.lr
        @assert length(b1.lw.windows.members) == length(b2.lw.windows.members)
        for (m1,m2) in zip(b1.lw.windows.members,b2.lw.windows.members)
            @assert m1==m2
        end
        @assert ntot1 == ntot2
        Ref(n1[]+n2[]),ntot1,ArrayBuffer(red.(b1.a,b2.a),b1.offsets,b1.lw)
    end
    OutputAggregator(o3,o1.bufsize)
end

function merge_all_outbuffers(outbuffers,red)
    @debug "Merging output buffers"
    reduce(outbuffers) do buf1, buf2
        res = merge_outbuffer_collection.(buf1,buf2,(red,))
    end
end

function flush_all_outbuffers(outbuffers,fin,outars,piddir)
    @assert length(outbuffers) == length(outars) == length(fin)
    for (coll,outar,f) in zip(outbuffers,outars,fin)
        allkeys = collect(keys(coll.buffers))
        for k in allkeys
            @debug "Writing index $k"
            put_buffer(k,f, last(coll.buffers[k]), coll, outar, piddir)
        end
    end
    GC.gc()
end
  
function generate_outbuffer_collection(ia,buftype,loopranges) 
    nd = getsubndims(ia)
    bufsize = getbufsize(ia,loopranges)
    d = Dict{BufferIndex{nd},Tuple{Base.RefValue{Int},Int,ArrayBuffer{Array{buftype,nd},NTuple{nd,Int},typeof(ia.lw)}}}()
    OutputAggregator(d,bufsize)
end

"Extracts or creates output buffer as an ArrayBuffer"
function extract_outbuffer(r,lr,outspecs,init,buftype,buffer::OutputAggregator)
    inds = get_bufferindices(r,outspecs)
    offsets = offset_from_range(inds)
    n,ntot,b = get!(buffer.buffers,inds) do 
        buf = generate_raw_outbuffer(init,buftype,buffer.bufsize)
        buf = ArrayBuffer(buf,offsets,outspecs.lw)
        ntot = bufferrepeat(r,lr,outspecs.lw)
        (Ref(0),ntot,buf)
    end
    n[] = n[]+1 
    b
end


"Check if maximum number of aggregations has happened for a buffer"
function mustwrite(inds,bufdict) 
    n_written,ntot,_ = bufdict.buffers[inds]
    if n_written[] > ntot
        error("Something is wrong, buffer got wrapped more often than it should. Make sure to use a runner only once")
    else
        n_written[] == ntot
    end
end

"Checks if output buffers have accumulated to the end and exports to output array"
function put_buffer(r, fin, bufnow, bufferdict, outar, piddir)
  bufinds = get_bufferindices(r,bufnow)
  if mustwrite(bufinds,bufferdict)
    offsets = offset_from_range(bufinds)
    i1 = first.(axes(outar))
    i2 = last.(axes(outar))
    inds = bufinds.indranges
    skip1 = max.(0,i1.-first.(inds))
    skip2 = max.(0,last.(inds).-i2)
    inds2 = range.(first.(inds).+skip1,last.(inds).-skip2)
    r2 = range.(1 .+ skip1, skip1 .+ length.(inds2))
    if piddir !== nothing
        @debug "$(myid()) acquiring lock $piddir to write to $inds2"
        mkpidlock(fetch(piddir),wait=true,stale_age=100) do
            broadcast!(fin,view(outar,inds2...),bufnow.a[r2...])
        end
    else
        broadcast!(fin,view(outar,inds2...),bufnow.a[r2...])
    end
    delete!(bufferdict.buffers,bufinds)
    true
  else
    false
  end
end

function create_buffers(inars, outars, f, loopranges)
    inbuffers_pure = generate_inbuffers(inars,loopranges)
    outbuffers_pure = generate_outbuffers(outars,f,loopranges)
    inbuffers_pure, outbuffers_pure
end